# Documentation

## Instructions to read the documentation

Open either the markdown ([Documentation.md](./Documentation.md)) or the pdf version ([Documentation.pdf](./Documentation.pdf)) of the documentation.

## Instructions to modify the documentation

First, install `quarto`:

```bash
pip install quarto
```

Then, modify [Documentation.qmd](./Documentation.qmd) under `_documentation` directory.

For live edit, execute:

```bash
quarto preview _documentation/Documentation.qmd --to pdf
```

Then, render into `pdf` and `md` files by executing:

```bash
quarto render _documentation/Documentation.qmd
```

Finally, push your changes.
