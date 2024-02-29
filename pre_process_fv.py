import json
import timeit
import pandas as pd
import psutil
import s3fs
import multiprocessing
import dask.bag as db
from dask.distributed import Client
from pyarrow import fs
import dask

dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
from spacer.data_classes import ImageFeatures
from spacer.messages import DataLocation


# Load selected_sources_labels parquet file
selected_sources_labels = pd.read_parquet(
    "pyspacer-test/allsource/selected_sources_labels.parquet", filesystem=fs
)
list_of_s3_urls = selected_sources_labels["key"].unique()

# bucket_name =
# prefix =

n_cores = multiprocessing.cpu_count()
print(n_cores)


total_memory = psutil.virtual_memory().total / (1024**3)  # Total memory in GB
print(total_memory)

memory_limit = total_memory / 2


with open("secrets.json", "r") as f:
    secrets = json.load(f)

# Create an S3FileSystem instance with the credentials from secrets.json
s3 = s3fs.S3FileSystem(
    key=secrets["AWS_ACCESS_KEY_ID"],
    secret=secrets["AWS_SECRET_ACCESS_KEY"],
    anon=False,
)
fs = fs.S3FileSystem(
    region=os.environ["S3_REGION"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
)


def load_featurevector(loc: DataLocation) -> pd.DataFrame:
    location = DataLocation("s3", bucket_name=bucket_name, key=loc)
    features = ImageFeatures.load(loc=location)
    data_for_df = [
        {
            "Column": feature_point.col,
            "Row": feature_point.row,
            "data": feature_point.data,
        }
        for feature_point in features.point_features
    ]

    df = pd.DataFrame(data_for_df)
    df["key"] = loc

    return df


def run_bag():
    # Start a Dask client
    # cluster = LocalCluster()
    client = Client(
        n_workers=n_cores,
        memory_limit=f"{memory_limit}GB",
        processes=False,
        threads_per_worker=2,
    )
    print(client.dashboard_link)

    # Create a Dask Bag from your list of S3 URLs
    urls = db.from_sequence(list_of_s3_urls)

    # Map the url_exists function onto the bag
    fvs = urls.map(load_featurevector)

    # Compute the results
    results = pd.concat(fvs, ignore_index=True)

    # Save to parquet
    results.to_parquet("feature_vectors.parquet")

    client.shutdown()
    return results


if __name__ == "__main__":
    # Time the run with timeit
    start = timeit.default_timer()
    dfs_up = run_bag()
    stop = timeit.default_timer()
    print("Time: ", stop - start)
