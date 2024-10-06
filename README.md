# Resume Ranking System

A Flask web application that processes and ranks resumes based on their relevance to a given job description. The application uses natural language processing (NLP) techniques to analyze resumes, 
extracting skills, years of experience, and sentiment to provide an overall ranking.

## Features

- **Resume Parsing**: Supports PDF and DOCX formats for resume submissions.
- **Skill Matching**: Compares the skills in resumes against the job description.
- **Experience Calculation**: Analyzes years of experience based on dates mentioned in resumes.
- **Sentiment Analysis**: Evaluates the sentiment of resumes and job descriptions.
- **Ranking System**: Ranks resumes based on similarity to the job description, experience, skills, and sentiment.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Flask
- Required libraries: `docx`, `pdfplumber`, `pandas`, `sklearn`, `nltk`, `textblob`, `langchain`

You can install the required libraries using:

```bash
pip install flask docx pdfplumber pandas scikit-learn nltk textblob langchain
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Saturnitty/lanchain_resume_parser.git
   cd lanchain_resume_parser
   ```

2. Set up your OpenAI API key in your environment variables:

   ```bash
   export OPENAI_API_KEY='your_api_key_here'  # Linux/Mac
   set OPENAI_API_KEY='your_api_key_here'     # Windows PowerShell
   ```

3. Download the necessary NLTK resources:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. Place your resumes in the `resumes` directory. This directory should contain all the resumes you want to process.

## Usage

To run the application, execute:

```bash
python app.py
```

The application will start a development server at `http://127.0.0.1:5000/`.

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Enter the job description in the provided input field.
3. Submit the form to see the ranked resumes based on the job description.

## Folder Structure

```plaintext
lanchain_resume_parser/
│
├── app.py                 # Main application file
├── requirements.txt       # List of dependencies
├── templates/             # HTML templates for rendering pages
│   ├── index.html         # Main input page
│   └── results.html       # Page to display results
├── resumes/               # Directory for storing resumes
│   ├── resume1.pdf
│   ├── resume2.docx
│   └── ...
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [NLTK](https://www.nltk.org/) for natural language processing.
- [LangChain](https://langchain.com/) for managing language model chains.
- [OpenAI](https://openai.com/) for the embeddings.

---

Feel free to reach out if you have any questions or need further assistance!
```

### How to Use This README
1. **Replace any placeholders** (like the OpenAI API key) with your actual values.
2. **Update the folder structure** if there are additional files or directories in your project.
3. **Add any additional sections** that you think might be useful, like troubleshooting or examples of input and output.

Let me know if you need any changes or additional information!
