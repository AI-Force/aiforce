import argparse
import ast
import glob
import itertools
import os
import pandas as pd
import shutil
from tqdm import tqdm
from ..image.opencv_tools import get_image_size, caculate_ssim, calculate_psnr, resize_image

DEFAULT_INDEX_FILE = "similarities.csv"
INDEX_PAIR_COLUMN = "pair"
INDEX_PSNR_COLUMN = "psnr"
INDEX_SSIM_COLUMN = "ssim"
INDEX_COLUMNS = [INDEX_PAIR_COLUMN, INDEX_PSNR_COLUMN, INDEX_SSIM_COLUMN]
PAIR_SEPARATOR = "|;|"


def create_dataframe(data=None):
    return pd.DataFrame(data, columns=INDEX_COLUMNS).set_index(INDEX_PAIR_COLUMN)


def load_dataframe(file_path):
    def convert_tuple(s):
        # handle inf conversion error by replacing with a big number
        s = s.replace('inf', '2e308')
        return ast.literal_eval(s)

    df = pd.read_csv(file_path, index_col=[INDEX_PAIR_COLUMN]).fillna("")
    df[INDEX_PSNR_COLUMN] = df[INDEX_PSNR_COLUMN].apply(convert_tuple)
    df[INDEX_SSIM_COLUMN] = df[INDEX_SSIM_COLUMN].apply(convert_tuple)
    return df


def save_dataframe(df, file_path):
    df.to_csv(file_path)


def split_pair_index(row):
    return row.name.split(PAIR_SEPARATOR)


def list_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True))
    input_dir_prefix = f"{input_dir}{os.path.sep}"
    files = list(map(lambda f: f.replace(input_dir_prefix, ""), files))
    return files


def create_similarity_index(file_a, file_b, mpsnr_thresh, mssim_thresh):
    img_a, w_a, h_a = get_image_size(file_a)
    img_b, w_b, h_b = get_image_size(file_b)
    min_height = min(h_a, h_b)
    min_width =  min(w_a, w_b)
    img_a_r = resize_image(img_a, min_width, min_height)
    img_b_r = resize_image(img_b, min_width, min_height)
    psnr = "" if mpsnr_thresh is None else calculate_psnr(img_a_r, img_b_r)
    mssism = "" if mssim_thresh is None else caculate_ssim(img_a_r, img_b_r)
    
    return [psnr, mssism]


def parse_index_file(files, input_dir, file_path, mpsnr_thresh, mssim_thresh):

    def filter_existing_files(row):
        file_a, file_b = row.name.split(PAIR_SEPARATOR)
        return os.path.isfile(os.path.join(input_dir, file_a)) and os.path.isfile(os.path.join(input_dir, file_b))

    def filter_empty_similarities(row):
        return (mpsnr_thresh is not None and row[INDEX_PSNR_COLUMN] == "") or (mssim_thresh is not None and row[INDEX_SSIM_COLUMN] == "")

    df = load_dataframe(file_path) if os.path.isfile(file_path) else create_dataframe()
    # filter existing files
    df = df[df.apply(filter_existing_files, axis=1)]

    pairs = list(itertools.combinations(files, 2))

    new_pairs = list(filter(lambda p: f"{p[0]}{PAIR_SEPARATOR}{p[1]}" not in df.index and f"{p[1]}{PAIR_SEPARATOR}{p[0]}" not in df.index, pairs))
    message = [] if mpsnr_thresh is None else [INDEX_PSNR_COLUMN]
    if mssim_thresh is not None:
        message.append(INDEX_SSIM_COLUMN)

    new_pairs_count = len(new_pairs)
    if (new_pairs_count):
        print(f"Add similarity index {' & '.join(message)} for {new_pairs_count} pairs to {file_path}. This may take a while...")

        new_similarities = []
        for file_a, file_b in tqdm(new_pairs):
            new_similarities.append([f"{file_a}{PAIR_SEPARATOR}{file_b}"] + create_similarity_index(os.path.join(input_dir, file_a),  os.path.join(input_dir, file_b), mpsnr_thresh, mssim_thresh))
        df = pd.concat([df, create_dataframe(new_similarities)])

    df_no_value = df[df.apply(filter_empty_similarities, axis=1)]
    df_no_value_count = len(df_no_value.index)
    if df_no_value_count:
        print(f"Calculate {' & '.join(message)} for {df_no_value_count} pairs at {file_path}. This may take a while...")
        for _, row in tqdm(df_no_value.iterrows(), total=df_no_value.shape[0]):
            file_a, file_b = row.name.split(PAIR_SEPARATOR)
            result = create_similarity_index(file_a, file_b, mpsnr_thresh, mssim_thresh)
            result[0] = row[INDEX_PSNR_COLUMN] if result[0] == "" else result[0]
            result[1] = row[INDEX_SSIM_COLUMN] if result[1] == "" else result[1]
            df.loc[row.name, [INDEX_PSNR_COLUMN, INDEX_SSIM_COLUMN]] = result

    return df


def collect_similar_images(df, input_dir, output_dir, mpsnr_thresh, mssim_thresh):
    similarities = {}

    def filter_similarities(row):
        return (mpsnr_thresh is not None and row[INDEX_PSNR_COLUMN][-1] > mpsnr_thresh) or (mssim_thresh is not None and row[INDEX_SSIM_COLUMN][-1] > mssim_thresh)

    df = df[df.apply(filter_similarities, axis=1)]
    num_similarities = len(df.index)
    print(f"Handle {num_similarities} similar pairs.")
    if num_similarities:
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            file_a, file_b = split_pair_index(row)
            similarities[file_b] = file_a if file_a not in similarities else similarities[file_a]
            target_folder = os.path.join(output_dir, os.path.basename(similarities[file_b]))
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(os.path.join(input_dir, file_a), target_folder)
            shutil.copy(os.path.join(input_dir, file_b), target_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="source directory",
                        type=str, default="input")
    parser.add_argument("-o", "--output_dir", help="target directory",
                        type=str, default="output")
    parser.add_argument("-s", "--similarity_index", help="file containing already calculated similarities",
                        type=str, default=None)
    parser.add_argument("-p", "--mpsnr", help="PSNR mean threshold to filter similar frames",
                        type=float, default=None)
    parser.add_argument("-m", "--mssim", help="SSIM mean threshold to filter similar frames",
                        type=float, default=None)

    args = parser.parse_args()

    print(os.getcwd())

    if not os.path.exists(args.input_dir):
        raise IOError("source directory does not exist")

    os.makedirs(args.output_dir, exist_ok=True)

    similarity_index_file = os.path.join(args.output_dir, DEFAULT_INDEX_FILE) if args.similarity_index is None else args.similarity_index

    print(f"Parse source directory {args.input_dir}")
    files = list_files(args.input_dir)

    similarities = parse_index_file(files, args.input_dir, similarity_index_file, args.mpsnr, args.mssim)

    target_file = os.path.join(args.output_dir, DEFAULT_INDEX_FILE)
    print(f"Write similarity index to {target_file}")
    save_dataframe(similarities, target_file)
    
    collect_similar_images(similarities, args.input_dir, args.output_dir, args.mpsnr, args.mssim)

    print(f"Finished !!!")
