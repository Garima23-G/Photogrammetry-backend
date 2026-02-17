DEFAULT_CATEGORIES = ("trench", "manhole", "duct", "handhole")


def normalize_category_name(category):
    if category is None:
        return ""
    return str(category).strip().lower()


def get_registered_categories():
    """
    Source of truth for supported categories.
    Extend this tuple as new classes are introduced.
    """
    return DEFAULT_CATEGORIES
