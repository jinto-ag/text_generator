# Text Generator

A package for generating text from questions/queries inputs using GPT-2 model.

## Installation

`
 
    pip install text_generator

`
## Usage
## Text Extraction

`
   
    from text_generator.text_extract import extract_text_from_pdf, get_pdf_files

    pdf_directory = "path/to/pdfs"
    pdf_files = get_pdf_files(pdf_directory)
    pdf_texts = extract_text_from_pdf(pdf_files)

`

## Text Generation
`

    from text_generator.text_gen import generate_text, load_model

    model = load_model()
    generated_text = generate_text(model, "What is the meaning of life?")

`

## Model Training

`

    from text_generator.text_gen import train_model

    model = load_model()
    optimized_model = optimize_model(model)
    trained_model = train_model(optimized_model,batch_size,epochs,max_length,dataset_path)

`
## Settings

All settings for the package can be found in text_generator/settings.py. This includes settings for the GPT-2 model, training, evaluation, pdf text extraction, and multithreading. These settings can be adjusted as needed for optimal performance.
Note

This package uses GPT-2 model, therefore it requires a large amount of memory to run. Make sure your machine has enough memory before running the package.
Contributing

This package is open to contributions, if you wish to contribute please fork the repository and submit a pull request.

## License

This package is open-source and available under the MIT License

## Features

- Multithreading support for pdf text extraction
- GPT-2 based text generation
- Global settings file for easy maintainability
- train, fine tune and load model functionality
- production ready package
- tests for all functionality

## TODO

- Add support for other text extraction libraries
- Add support for other models for text generation

This README.md file gives an overview of how to use the package, how to train the model, how to extract pdf files and also other features of the package. It also