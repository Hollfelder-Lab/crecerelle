from urllib.parse import quote


REPOSITORY = "Hollfelder-Lab/crecerelle"
BRANCH = "main"


def on_page_content(html, page, config, files):
    src_path = page.file.src_path.replace("\\", "/")
    if not src_path.endswith(".ipynb"):
        return html

    quoted_path = quote(src_path, safe="/")
    colab_url = (
        "https://colab.research.google.com/github/"
        f"{REPOSITORY}/blob/{BRANCH}/{quoted_path}"
    )
    button = (
        '<p><a class="md-button md-button--primary" '
        f'href="{colab_url}" target="_blank" rel="noopener">'
        "Open in Colab</a></p>"
    )
    return f"{button}\n{html}"
