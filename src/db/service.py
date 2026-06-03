from typing import List, Optional, Union

from loguru import logger
from sqlalchemy import select

from src.db.db import get_db_session
from src.db.models import Product, ProductEmbedding, Review, Task
from src.models.enums import TaskStatus
from src.models.schemas import ProductEmbeddingCreate, ProductCreate, ReviewCreate, ProductRead, TaskUpdate, \
    ProductEmbeddingRead, TaskRead


class ProductService:
    @staticmethod
    async def insert_product(
        product_create: ProductCreate
    ) -> int:
        async with get_db_session() as db:
            product = Product.model_validate(product_create.model_dump())
            db.add(product)

            await db.flush()

            return product.id

    @staticmethod
    async def get_product(
        product_id: int,
    ):
        async with get_db_session() as db:
            result = await db.execute(
                select(Product).where(Product.id == product_id)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def get_products(
        product_ids: List[int],
    ) -> List[ProductRead]:
        if not product_ids:
            return []

        async with get_db_session() as db:
            result = await db.execute(
                select(Product).where(Product.id.in_(product_ids))
            )
            db_products = result.scalars().all()

            return [
                ProductRead.model_validate(db_product)
                for db_product in db_products
            ]

    @staticmethod
    async def insert_products(
        products_create: List[ProductCreate]
    ) -> List[int]:
        async with get_db_session() as db:
            products = [
                Product.model_validate(p.model_dump())
                for p in products_create
            ]

            db.add_all(products)
            await db.flush()

            return [product.id for product in products]

    @staticmethod
    async def insert_products_with_embeddings_and_reviews(
        products_create: List[ProductCreate],
        embeddings_create: List[ProductEmbeddingCreate],
        reviews_create: List[List[ReviewCreate]]  # List of lists for matching products
    ) -> List[int]:
        async with get_db_session() as db:
            products_to_add = []
            for p_data, emb_data, product_reviews in zip(
                    products_create, embeddings_create, reviews_create
            ):
                product = Product.model_validate(p_data.model_dump())
                embedding = ProductEmbedding.model_validate(emb_data.model_dump())
                product.embedding = embedding

                # Instantiate and attach the One-to-Many Reviews
                for r_data in product_reviews:
                    review = Review.model_validate(r_data.model_dump())
                    product.reviews.append(review)

                products_to_add.append(product)

            db.add_all(products_to_add)

            # Flush to execute the inserts and populate the IDs
            await db.flush()

            return [product.id for product in products_to_add]

    @staticmethod
    async def insert_product_embeddings(
        product_embeddings: List[ProductEmbeddingCreate]
    ) -> None:

        db_product_embeddings = [ProductEmbedding.model_validate(product_embedding.model_dump())
                                 for product_embedding in product_embeddings]

        async with get_db_session() as db:
            db.add_all(db_product_embeddings)

    @staticmethod
    async def insert_product_reviews(
        product_reviews: List[ReviewCreate]
    ) -> None:

        db_product_reviews = [Review.model_validate(product_review.model_dump())
                                 for product_review in product_reviews]

        async with get_db_session() as db:
            db.add_all(db_product_reviews)

    @staticmethod
    async def get_all_product_urls() -> set[str]:
        async with get_db_session() as db:
            result = await db.execute(
                select(Product.url)
            )
            # scalars() flattens the tuple results into single elements
            return set(result.scalars().all())

    @staticmethod
    async def get_all_product_emb_info() -> List[ProductEmbeddingRead]:
        async with get_db_session() as db:
            stmt = select(
                ProductEmbedding.product_id,
                ProductEmbedding.name,
                ProductEmbedding.generated_description
            ).distinct()

            result = await db.execute(stmt)

            rows = result.mappings().all()

            return [ProductEmbeddingRead.model_validate(row) for row in rows]

    @staticmethod
    async def get_all_embeddings() -> List[dict]:
        async with get_db_session() as db:
            result = await db.execute(
                select(
                    ProductEmbedding.id,
                    ProductEmbedding.name,
                    ProductEmbedding.structured_embedding,
                    ProductEmbedding.generated_description_embedding,
                    Product.category_name
                )
                .join(Product)
            )
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "structured_embedding": row.structured_embedding,
                    "generated_embedding": row.generated_description_embedding,
                    "category_name": row.category_name
                }
                for row in result.all()
            ]

    @staticmethod
    async def get_similarity_search_query(embedding_vector: list[float], limit: int = 25) -> List[ProductEmbedding]:
        """
        Returns top_k documents using pgvector cosine distance (<=>).
        """
        async with get_db_session() as db:
            result = await db.execute(
                select(ProductEmbedding)
                .order_by(ProductEmbedding.generated_description_embedding.cosine_distance(embedding_vector))
                .limit(limit)
            )

            return result.scalars().all()


class TaskService:
    @staticmethod
    async def insert_task(user_query: str, callback_url: str, task_uuid: str):
        async with get_db_session() as db:
            new_task = Task(
                task_uuid=task_uuid,
                status=TaskStatus.waiting.value,
                user_query=user_query,
                callback_url=callback_url,
                llm_result_text=None
            )
            db.add(new_task)
            await db.flush()

    @staticmethod
    async def get_task(task_uuid: str) -> Union[TaskRead, None]:
        async with get_db_session() as db:
            result = await db.execute(
                select(Task).where(Task.task_uuid == task_uuid)
            )
            db_task = result.scalar_one_or_none()

            task_data = None
            if not db_task:
                logger.error(f"Task with UUID {task_uuid} not found in database.")
            else:
                task_data = TaskRead.model_validate(db_task)

            return task_data

    @staticmethod
    async def update_task(task_uuid: str, update_data: TaskUpdate) -> Optional[Task]:
        async with get_db_session() as db:
            result = await db.execute(
                select(Task).where(Task.task_uuid == task_uuid)
            )
            db_task = result.scalar_one_or_none()

            if not db_task:
                logger.error(f"Task with UUID {task_uuid} not found for database update operation.")
                return None

            update_dict = update_data.model_dump(exclude_unset=True)
            db_task.sqlmodel_update(update_dict)

            db.add(db_task)
            await db.flush()

            logger.info(f"Database row updated for Task UUID: {task_uuid} | Status changed to: {db_task.status}")
            return db_task
