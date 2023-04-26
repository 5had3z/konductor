# Konducting Reviews

This can be extended with more custom pages for your particular scenario by creating a new launcher and adding more pages before launching. These will then appear in the top button group. You will also need to now launch with this script, rather than konduct-review.

```python
import dash
from konductor.webserver.app import app, get_basic_layout, add_base_args
import custom_page

dash.register_page(
    "my-custom-page", path="/my-custom-layout", layout=custom_page.layout
)

if __name__ == "__main__":
    parser = ArgumentParser()
    add_base_args(parser)
    app.layout = get_basic_layout(str(parser.parse_args().root))
    app.run(debug=True)
```

## TODO
 - Figure out how to get option tree/experiments in one global location, rather than sharing the root dir and recreating those in the other pages.
 - Make component names prefixed with something unique so there's less chance of collisions with 3rd party pages.
