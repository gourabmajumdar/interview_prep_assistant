# 🎯 AI-Powered Interview Preparation Assistant

An advanced interview preparation tool that analyzes job descriptions and generates personalized interview questions with comprehensive answers using AI and real-time web search.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Intelligent Job Analysis
- Extracts technical skills, experience requirements, and soft skills from job descriptions
- Identifies role level (Junior/Mid/Senior) and education requirements
- Analyzes company-specific information

### Multi-Source Web Search
- **Glassdoor**: Real interview experiences and questions
- **Indeed**: Interview tips and guides  
- **LeetCode**: Coding problems for technical roles
- **GitHub**: Curated interview resources
- **Reddit**: Community discussions
- **Google**: General search results

### AI-Powered Question Generation
- **Claude (Opus 4 & Sonnet 4)**: Anthropic's latest models for nuanced technical questions
- **GPT-4**: OpenAI's advanced model for comprehensive answers
- Generates 15-30 customized questions per job description
- Includes detailed answers with code examples where relevant

### Comprehensive Question Coverage
- Technical skills assessment
- System design (for senior roles)
- Behavioral questions using STAR method
- Company-specific questions
- Coding problems
- Culture fit questions

### Export & Reporting
- Download interview prep guide as PDF/Markdown
- Export questions to JSON format
- Track preparation progress
- Save personalized study materials

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- API keys for either Anthropic (Claude) or OpenAI (GPT-4)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/gourabmajumdar/interview_prep_assistant.git
cd interview_prep_assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys**

Create a `.env` file in the project root:
```env
# At least one of these is required
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

### Running the Application

```bash
streamlit run interview_assistant.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Configure Settings (Sidebar)
- Select your preferred AI model (Claude or GPT-4)
- Choose web search sources
- Set number of questions to generate
- Configure other preferences

### 2. Analyze Job Description
- Paste the complete job description
- Add company name and role title
- Click "Analyze & Search"
- Review extracted skills and requirements

### 3. Generate Interview Questions
- Navigate to "Interview Questions" tab
- Click "Generate Interview Questions"
- Review personalized questions with:
  - Comprehensive answers
  - Preparation tips
  - Difficulty levels
  - Source attribution

### 4. Export and Practice
- Download as PDF for offline study
- Export to JSON for custom applications
- Use the preparation checklist
- Access additional resources

## 🔧 Configuration

### API Keys Setup

#### Anthropic (Claude)
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key in Account Settings
3. Add credits to your account ($5 minimum)

#### OpenAI (GPT-4)
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Generate API key in API Keys section
3. Add billing information

### Cost Estimates
- Light usage (10-20 analyses): ~$2-5/month
- Moderate usage (50-100 analyses): ~$10-20/month
- Heavy usage (200+ analyses): ~$30-50/month

## 🏗️ Project Structure

```
interview-prep-assistant/
├── interview_assistant.py    # Main application
├── requirements.txt         # Python dependencies
├── .env                    # API keys (create this)
├── .gitignore             # Git ignore file
├── README.md              # This file
└── assets/                # Screenshots and resources
```

## 📦 Dependencies

- **streamlit**: Web application framework
- **anthropic**: Claude API client
- **openai**: GPT-4 API client (v0.28.1)
- **duckduckgo-search**: Web search functionality
- **beautifulsoup4**: Web scraping
- **selenium**: Advanced web content extraction
- **aiohttp**: Async HTTP requests
- **python-dotenv**: Environment variable management

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **API Key Errors**
   - Verify API keys are correctly set in `.env` file
   - Check if you have credits/billing set up

3. **No Questions Generated**
   - Ensure job description is detailed enough
   - Check internet connection for web search
   - Verify AI model is selected in sidebar

4. **JSON Parsing Errors**
   - Try switching between Claude and GPT-4
   - Check if API rate limits are hit

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- OpenAI for GPT-4 API
- Streamlit for the web framework
- The open-source community for various libraries

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

## 🚀 Future Enhancements

- [ ] Voice-based mock interview practice
- [ ] Integration with calendar for interview scheduling  
- [ ] Progress tracking and analytics
- [ ] Mobile application
- [ ] Integration with more job boards
- [ ] Multi-language support
- [ ] Video interview preparation tips
- [ ] Integration with LinkedIn for profile optimization

---

Made with ❤️ for job seekers everywhere. Good luck with your interviews! 🎉