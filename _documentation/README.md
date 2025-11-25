# Documentation

## Instructions to read the documentation

Open either the markdown ([Documentation.md](./Documentation.md)) or the pdf version ([Documentation.pdf](./Documentation.pdf)) of the documentation.

## Instructions to modify the documentation

First, install `quarto`:

```bash
pip install quarto
```

Then, modify [Documentation.qmd](./authoring/Documentation.qmd) under `_documentation/authoring` directory.

For live edit, execute:

```bash
quarto preview _documentation/authoring/Documentation.qmd --to pdf
```

Then, render into `pdf` and `md` files by executing:

```bash
quarto render _documentation/authoring/Documentation.qmd --output-dir ..
```

Finally, push your changes.
