import io

from miniopy_async import Minio
import asyncio
import numpy as np
from PIL import Image


# https://github.com/hlf20010508/miniopy-async/tree/master/examples/simple_examples

async def main():
    async with Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="admin123",
        secure=False,  # http for False, https for True
    ) as client:
        bucket_name = "request-files"  # !! can't use underscores "_"
        task_file_name = "request_file_31bf1g21"

        result = await client.bucket_exists(bucket_name)
        if result:
            print(f"{bucket_name} exists")
        else:
            print(f"{bucket_name} does not exist")
            print("Creating bucket")
            await client.make_bucket(bucket_name)

        # load file
        with open("image.png", "rb") as f:
            # Converts bytes into an in-memory binary stream
            binary_stream = io.BytesIO(f.read())

        print(type(binary_stream))
        stream_length = binary_stream.getbuffer().nbytes
        print(f"Stream type: {type(binary_stream)}, Length: {stream_length} bytes")

        # Upload file
        await client.put_object(
            bucket_name=bucket_name,
            object_name=task_file_name,
            data=binary_stream,
            length=stream_length
        )

        # Check files
        print(f"Checking uploaded files in bucket {bucket_name}")
        objects = await client.list_objects(bucket_name)
        for obj in objects:
            print("obj:", obj)

        # Get file data
        response = await client.get_object(
            bucket_name=bucket_name,
            object_name=task_file_name
        )
        # Read the entire stream from the response into memory as bytes
        img_bytes = await response.content.read()

        # Convert bytes to an in-memory binary stream
        img_stream = io.BytesIO(img_bytes)

        # Open the image with PIL and convert to a NumPy array
        pil_image = Image.open(img_stream).convert("RGB")
        image_array = np.array(pil_image)

        # Verify the shape and type for your model
        print("NumPy Array Shape:", image_array.shape)
        print("Data Type:", image_array.dtype)

        pil_image.save(fp="./image_pil_get.png")

asyncio.run(main())
