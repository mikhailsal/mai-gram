"""Entry point for mAI Companion."""

import logging


def main() -> None:
    """Start the mAI Companion service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("mAI Companion starting up...")
    logger.info("Service not yet implemented. See the implementation plan for details.")


if __name__ == "__main__":
    main()
