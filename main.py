import argparse
import json
import os
from copy import copy

import cv2
import imagehash as imagehash
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.metrics import structural_similarity as compare_ssim, normalized_root_mse
import pandas
def crop(img):
    blurred = cv2.blur(img, (3, 3))
    canny = cv2.Canny(blurred, 50, 200)

    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    return img[y1:y2, x1:x2]

def main(folder1_path,folder2_path,export_name,show_images,debug=False):
    # Get a list of files in each folder
    folder1_file_names = set(os.listdir(folder1_path))
    folder2_file_names = set(os.listdir(folder2_path))
    file_names = folder1_file_names.intersection(folder2_file_names)
    rows = []
    for i,file_name in enumerate(file_names):
        try:
            #if i > 10: break
            print(f"{i+1}/{len(file_names)} {file_name}")


            image_1 = f"{folder1_path}/{file_name}"
            image_2 = f"{folder2_path}/{file_name}"
            # Load the images
            image1 = io.imread(image_1)
            image2 = io.imread(image_2)
            image1_gray = color.rgb2gray(image1)
            image2_gray = color.rgb2gray(image2)

            image1_resized = copy(image1)
            image2_resized = copy(image2)
            if (image1_resized.shape[1] * image1_resized.shape[0]) > (image2_resized.shape[1] * image2_resized.shape[0]):
                image1_resized = cv2.resize(image1_resized, (image2_resized.shape[1], image2_resized.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                image2_resized = cv2.resize(image2_resized, (image1_resized.shape[1], image1_resized.shape[0]), interpolation=cv2.INTER_AREA)
            image1_resized_gray = color.rgb2gray(image1_resized)
            image2_resized_gray = color.rgb2gray(image2_resized)

            image1_cropped_resized = crop(copy(image1))
            image2_cropped_resized = crop(copy(image2))
            if (image1_cropped_resized.shape[1] * image1_cropped_resized.shape[0]) > (image2_cropped_resized.shape[1] * image2_cropped_resized.shape[0]):
                image1_cropped_resized = cv2.resize(image1_cropped_resized, (image2_cropped_resized.shape[1], image2_cropped_resized.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                image2_cropped_resized = cv2.resize(image2_cropped_resized, (image1_cropped_resized.shape[1], image1_cropped_resized.shape[0]), interpolation=cv2.INTER_AREA)
            image1_cropped_resized_gray = color.rgb2gray(image1_cropped_resized)
            image2_cropped_resized_gray = color.rgb2gray(image2_cropped_resized)



            # Calculate the SSIM index
            ssim_index_grayscale_results = compare_ssim(image1_resized_gray, image2_resized_gray, data_range=max(image2_resized_gray.max(),image1_resized_gray.max()) - min(image2_resized_gray.min(),image1_resized_gray.min()))
            ssim_index_results = compare_ssim(image1_resized, image2_resized, multichannel=True, win_size=min(min(image1_resized.shape),min(image2_resized.shape)))
            cropped_ssim_index_grayscale_results = compare_ssim(image1_cropped_resized_gray, image2_cropped_resized_gray, data_range=max(image2_cropped_resized_gray.max(),image1_cropped_resized_gray.max()) - min(image2_cropped_resized_gray.min(),image1_cropped_resized_gray.min()))
            cropped_ssim_index_results = compare_ssim(image1_cropped_resized, image2_cropped_resized, multichannel=True, win_size=min(min(image1_cropped_resized.shape),min(image2_cropped_resized.shape)))

            # Calculate the Normalized RMSE
            nrmse_grayscale_results = 1 - normalized_root_mse(image1_resized_gray, image2_resized_gray,normalization="min-max")
            rmse_results = 1 - normalized_root_mse(image1_resized, image2_resized, normalization="min-max")
            cropped_nrmse_grayscale_results = 1 - normalized_root_mse(image1_cropped_resized_gray, image2_cropped_resized_gray,normalization="min-max")
            cropped_rmse_results = 1 - normalized_root_mse(image1_cropped_resized, image2_cropped_resized, normalization="min-max")

            # Compute the perceptual hashes
            image1_resized_hash = imagehash.phash(Image.fromarray(image1_resized))
            image2_resized_hash = imagehash.phash(Image.fromarray(image2_resized))
            image1_cropped_hash = imagehash.phash(Image.fromarray(image1_cropped_resized))
            image2_cropped_hash = imagehash.phash(Image.fromarray(image2_cropped_resized))
            image1_resized_gray_hash = imagehash.phash(Image.fromarray(image1_resized_gray))
            image2_resized_gray_hash = imagehash.phash(Image.fromarray(image2_resized_gray))
            image1_cropped_gray_hash = imagehash.phash(Image.fromarray(image1_cropped_resized_gray))
            image2_cropped_gray_hash = imagehash.phash(Image.fromarray(image2_cropped_resized_gray))
            hash_results = max(1.0 - (image1_resized_hash - image2_resized_hash) / max(len(image1_resized_hash.hash), len(image2_resized_hash.hash)),0.0)
            cropped_hash_results = max(1.0 - (image1_cropped_hash - image2_cropped_hash) / max(len(image1_cropped_hash.hash), len(image2_cropped_hash.hash)),0.0)
            hash_grayscale_results = max(1.0 - (image1_resized_gray_hash - image2_resized_gray_hash) / max(len(image1_resized_gray_hash.hash), len(image2_resized_gray_hash.hash)),0.0)
            cropped_hash_grayscale_results = max(1.0 - (image1_cropped_gray_hash - image2_cropped_gray_hash) / max(len(image1_cropped_hash.hash), len(image2_cropped_hash.hash)),0.0)



            # Apply template matching, get the maximum correlation value and its location.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.matchTemplate(image2_resized, image1_resized, cv2.TM_CCOEFF_NORMED))

            # Convert the correlation value to a percentage
            template_matching_result = (max_val + 1)/2  # Scale from -1 to 1 to 0 to 100

            row = {
                "name":file_name,
                "ssim_index_grayscale": ssim_index_grayscale_results,
                "ssim_index_colored": ssim_index_results,
                "normalized_rmse_index_grayscale":nrmse_grayscale_results,
                "normalized_rmse_index_colored":rmse_results,
                "hash_similarity_colored":hash_results,
                "template_matching_colored":template_matching_result,
                "ssim_index_grayscale_cropped":cropped_ssim_index_grayscale_results,
                "ssim_index_colored_cropped":cropped_ssim_index_results,
                "normalized_rmse_index_grayscale_cropped":cropped_nrmse_grayscale_results,
                "normalized_rmse_index_colored_cropped":cropped_rmse_results,
                "hash_similarity_colored_cropped":cropped_hash_results
            }
            rows.append(row)
            if debug:
                with open(f"last_result.json", 'w', encoding='utf-8') as file:
                    json.dump(row, file, ensure_ascii=False, indent=4)
            if show_images:
                cv2.imshow("image_1", image1_resized)
                cv2.imshow("image_2", image2_resized)
                cv2.imshow("image_1_cropped", image1_cropped_resized)
                cv2.imshow("image_2_cropped", image2_cropped_resized)
                cv2.waitKey(0)
        except Exception as e:
            print(e)
    if debug:
        with open(f"{export_name}.json", 'w', encoding='utf-8') as file:
            json.dump(rows, file, ensure_ascii=False, indent=4)
    df = pandas.DataFrame(rows)
    df.to_excel(f"{export_name}.xlsx", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some attributes.')
    parser.add_argument('--original_folder', type=str, nargs='?', default="./img_originals", help='Folder where the original images are. If not specified, default value of ./img_originals will be used.')
    parser.add_argument('--scraped_folder', type=str, nargs='?', default="./img", help='Folder where the scraped images are. If not specified, default value of ./img will be used.')
    parser.add_argument('--export_name', type=str, nargs='?', default="results",help='Results will be written to a .XLSX file under this name. Default value is "results".')
    parser.add_argument('--dont_show_images', action='store_true', help='For each image comparison, it will show both images and wait until any keyboard button to continue.')
    parser.add_argument('--debug', action='store_true', help='Enables debug. Debug means more printing to the terminal and also exports lots of redundant json files.')

    args = parser.parse_args()
    main(folder1_path=args.original_folder,folder2_path=args.scraped_folder,export_name=args.export_name,show_images=not args.dont_show_images,debug=args.debug)