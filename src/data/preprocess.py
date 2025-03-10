def clean_context(context: str) -> str:
    """
    Clean the context by removing unnecessary characters and formatting.
    """
    # Remove leading and trailing whitespaces
    context = context.strip()

    # Remove &#x0D;
    context = context.replace("&#x0D;", "")

    # Remove emty lines
    context = "\n".join([line for line in context.splitlines() if line.strip()])

    return context
