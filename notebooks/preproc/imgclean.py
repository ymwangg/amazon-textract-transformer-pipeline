# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker Processing script to prepare raw documents into SMGT-ready page images
"""

# Python Built-Ins:
import argparse
import logging
from multiprocessing import cpu_count, Pool
import os
import shutil
import time
from typing import Iterable, List, Optional

logging.basicConfig(level="INFO", format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

# External Dependencies:
import pdf2image
import PIL
from PIL import ExifTags


logger = logging.getLogger("preproc")


def get_exif_tag_id_by_name(name: str) -> Optional[str]:
    """Find a numeric EXIF tag ID by common name

    As per https://pillow.readthedocs.io/en/stable/reference/ExifTags.html
    """
    try:
        return next(k for k in ExifTags.TAGS.keys() if ExifTags.TAGS[k] == name)
    except StopIteration:
        return None


ORIENTATION_EXIF_ID = get_exif_tag_id_by_name("Orientation")


def split_filename(filename):
    basename, _, ext = filename.rpartition(".")
    return basename, ext


class ImageExtractionResult:
    """Result descriptor for extracting a source image/document to image(s)"""

    def __init__(self, rawpath: str, cleanpaths: List[str] = [], cats: List[str] = []):
        self.rawpath = rawpath
        self.cleanpaths = cleanpaths
        self.cats = cats


def clean_document_for_img_ocr(
    rel_filepath: str,
    from_basepath: str,
    to_basepath: str,
    pdf_dpi: int = 300,
    pdf_image_format: str = "png",
    allowed_formats: Iterable[str] = ("jpg", "jpeg", "png"),
    preferred_image_format: str = "png",
    wait: float = 0,
) -> ImageExtractionResult:
    """Process an individual PDF or image for use with SageMaker Ground Truth image task UIs

    Extracts page images from PDFs, converts EXIF-rotated images to data-rotated.

    Parameters
    ----------
    rel_filepath : str
        Path (relative to from_basepath) to the raw document/image to be processed
    from_basepath : str
        Base input path which should be masked for mapping to `to_basepath`
    to_basepath : str
        Target path for converted files (subfolder structure will be preserved from input)
    pdf_dpi : int
        DPI resolution to extract images from PDFs (Default 300).
    pdf_image_format : str
        Format to extract images from PDFs (Default 'png').
    allowed_formats : Iterable[str]
        The set of permitted file formats for compatibility: Used to determine whether to convert
        source images in other formats which PIL may still have been able to successfully load.
        NOTE: Amazon Textract also supports 'tiff', but we left it out of the default list because
        TIFF images seemed to break the SageMaker Ground Truth built-in bounding box UI as of some
        tests in 2021-10. Default ('jpg', 'jpeg', 'png').
    preferred_image_format : str
        Format to be used when an image has been saved/converted (Default 'png').
    wait : float
        If !=0, this function will `time.sleep(wait)` before running. This is useful for batch jobs
        utilizing all available cores on a machine, where errors might arise if we don't provide a
        little room for any background processes.
    """
    if wait:
        time.sleep(wait)

    full_filepath = os.path.join(from_basepath, rel_filepath)
    filename = os.path.basename(rel_filepath)
    subfolder = os.path.dirname(rel_filepath)
    outfolder = os.path.join(to_basepath, subfolder)
    os.makedirs(outfolder, exist_ok=True)
    basename, ext = split_filename(filename)
    ext_lower = ext.lower()
    result = ImageExtractionResult(
        rawpath=full_filepath,
        cleanpaths=[],
        cats=subfolder[1:].split(os.path.sep),  # Strip leading slash to avoid initial ''
    )
    if ext_lower == "pdf":
        logger.info(
            "Converting {} to {}/{}*.{}".format(
                full_filepath,
                outfolder,
                basename + "-",
                pdf_image_format,
            )
        )
        images = pdf2image.convert_from_path(
            full_filepath,
            output_folder=outfolder,
            output_file=basename + "-",
            # TODO: Use paths_only option to return paths instead of image objs
            fmt=pdf_image_format,
            dpi=pdf_dpi,
        )
        result.cleanpaths = [i.filename for i in images]
        logger.info(
            "* PDF converted {}:\n    - {}".format(
                full_filepath,
                "\n    - ".join(result.cleanpaths),
            )
        )
        return result

    try:
        image = PIL.Image.open(full_filepath)
    except PIL.UnidentifiedImageError:
        logger.warning(f"* Ignoring incompatible file: {full_filepath}")
        return None

    # Some "image" formats (notably TIFF) support multiple pages as "frames":
    n_image_pages = getattr(image, "n_frames", 1)
    if n_image_pages > 1:
        logger.info("Extracting %s pages from file %s", n_image_pages, full_filepath)
    convert_format = not ext_lower in allowed_formats
    outpaths = []
    for ixpage in range(n_image_pages):
        if n_image_pages > 1:
            image.seek(ixpage)

        # Correct orientation from EXIF data:
        exif = dict((image.getexif() or {}).items())
        img_orientation = exif.get(ORIENTATION_EXIF_ID)
        if img_orientation == 3:
            image = image.rotate(180, expand=True)
            rotated = True
        elif img_orientation == 6:
            image = image.rotate(270, expand=True)
            rotated = True
        elif img_orientation == 8:
            image = image.rotate(90, expand=True)
            rotated = True
        else:
            rotated = False

        if n_image_pages == 1 and not (convert_format or rotated):
            # Special case where image file can just be copied across:
            outpath = os.path.join(outfolder, filename)
            shutil.copy2(full_filepath, outpath)
        else:
            outpath = os.path.join(
                outfolder,
                "".join(
                    (
                        basename,
                        "-%04i" % (ixpage + 1) if n_image_pages > 1 else "",
                        ".",
                        preferred_image_format if convert_format else ext,
                    )
                ),
            )
            image.save(outpath)

        outpaths.append(outpath)
        logger.info(
            "* %s image %s%s (orientation %s) to %s",
            "Converted"
            if convert_format
            else ("Rotated" if rotated else ("Extracted" if n_image_pages > 1 else "Copied")),
            full_filepath,
            f" page {ixpage + 1}" if n_image_pages > 1 else "",
            img_orientation,
            outpath,
        )

    result.cleanpaths = outpaths
    return result


def clean_dataset_for_img_ocr(
    from_path: str,
    to_path: str,
    filepaths: Optional[Iterable[str]] = None,
    pdf_dpi: int = 300,
    pdf_image_format: str = "png",
    allowed_formats: Iterable[str] = ("jpg", "jpeg", "png"),
    preferred_image_format: str = "png",
) -> List[ImageExtractionResult]:
    """Process a mixed PDF/image dataset for use with SageMaker Ground Truth image task UIs

    This is a convenience method for processing documents directly in notebook, and is not used
    when run as a processing job. Extracts page images from PDFs, converts EXIF-rotated images to
    data-rotated.

    Parameters
    ----------
    from_path : str
        Base path of the raw/source dataset to be converted
    to_path : str
        Target path for converted files (subfolder structure will be preserved from source)
    filepaths : Optional[Iterable[str]]
        Paths (relative to from_path, no leading slash) to filter down the processing. If not
        provided, the whole from_path folder will be recursively crawled.
    pdf_dpi : int
        DPI resolution to extract images from PDFs (Default 300).
    pdf_image_format : str
        Format to extract images from PDFs (Default 'png').
    allowed_formats : Iterable[str]
        The set of permitted file formats for compatibility: Used to determine whether to convert
        source images in other formats which PIL may still have been able to successfully load.
        NOTE: Amazon Textract also supports 'tiff', but we left it out of the default list because
        TIFF images seemed to break the SageMaker Ground Truth built-in bounding box UI as of some
        tests in 2021-10. Default ('jpg', 'jpeg', 'png').
    preferred_image_format : str
        Format to be used when an image has been saved/converted (Default 'png').
    """
    results = []
    if not filepaths:
        # List input files automatically by walking the from_path dir:
        filepaths = [
            path[len(from_path) + 1 :]  # Convert to relative path (+1 for trailing '/')
            for path in filter(
                lambda path: "/." not in path,  # Ignore hidden stuff e.g. .ipynb_checkpoints
                (os.path.join(path, f) for path, _, files in os.walk(from_path) for f in files),
            )
        ]
    os.makedirs(to_path, exist_ok=True)

    for ixfilepath, filepath in enumerate(filepaths):
        logger.info("Doc %s of %s", ixfilepath + 1, len(filepaths))
        result = clean_document_for_img_ocr(
            filepath,
            from_path,
            to_path,
            pdf_dpi=pdf_dpi,
            pdf_image_format=pdf_image_format,
            allowed_formats=allowed_formats,
            preferred_image_format=preferred_image_format,
            wait=0,  # No need to pause in single-threaded for loop
        )
        if result:
            results.append(result)
        # Otherwise doc was skipped and warning already printed

    logger.info("Done!")
    return results


def parse_args():
    """Parse SageMaker Processing Job CLI arguments to job parameters"""
    parser = argparse.ArgumentParser(
        description="Split and standardize documents into images for OCR"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/opt/ml/processing/input/raw",
        help="Folder where raw input images/documents are stored",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/ml/processing/output/imgs-clean",
        help="Folder where cleaned output images should be saved",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use for extraction (default #CPU cores)",
    )
    args = parser.parse_args()
    return args


def process_doc_in_worker(inputs: dict):
    return clean_document_for_img_ocr(
        rel_filepath=inputs["rel_filepath"],
        from_basepath=inputs["in_folder"],
        to_basepath=inputs["out_folder"],
        wait=0.5,
    )


if __name__ == "__main__":
    # Main processing job script:
    args = parse_args()

    logger.info("Reading raw files from %s", args.input)
    rel_filepaths_all = sorted(
        filter(
            lambda f: not (f.startswith(".") or "/." in f),  # (Exclude hidden dot-files)
            [
                os.path.join(currpath, name)[len(args.input) + 1 :]  # +1 for trailing '/'
                for currpath, dirs, files in os.walk(args.input)
                for name in files
            ],
        )
    )

    n_docs = len(rel_filepaths_all)
    logger.info("Processing %s files across %s processes", n_docs, args.n_workers)
    with Pool(args.n_workers) as pool:
        for ix, result in enumerate(
            pool.imap_unordered(
                process_doc_in_worker,
                [
                    {
                        "in_folder": args.input,
                        "out_folder": args.output,
                        "rel_filepath": path,
                    }
                    for path in rel_filepaths_all
                ],
            )
        ):
            logger.info("Processed doc %s of %s", ix + 1, n_docs)
    logger.info("All done!")
