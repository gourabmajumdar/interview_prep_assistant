"""
Advanced Interview Preparation Assistant
Real-time web search and advanced AI integration for comprehensive interview preparation

Requirements:
pip install streamlit anthropic openai google-api-python-client duckduckgo-search \
           beautifulsoup4 pandas python-dotenv aiohttp asyncio selenium webdriver-manager \
           langchain chromadb tiktoken pydantic tenacity markdown2 pypdf
"""

import streamlit as st
import os
import json
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential

# AI Model Imports
import anthropic
import openai
from duckduckgo_search import DDGS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Load environment variables
load_dotenv()


# Configuration
class Config:
    """Configuration settings for AI models and search"""
    # AI Model API Keys
    try:
        # Try to get from Streamlit secrets (for deployed app)
        ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    except:
        # Fall back to environment variables (for local development)
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Search Configuration
    SERP_API_KEY = os.getenv("SERP_API_KEY", "")  # For Google Search
    SEARCH_TIMEOUT = 30
    MAX_SEARCH_RESULTS = 10

    # Cache Configuration
    CACHE_DURATION = 3600  # 1 hour
    CACHE_DIR = "interview_cache"

    # AI Model Selection
    DEFAULT_AI_MODEL = "claude-opus-4"  # or "gpt-4", "claude-sonnet-4"

    # Rate Limiting
    RATE_LIMIT_DELAY = 1  # seconds between API calls


class AIModel(Enum):
    """Available AI models"""
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"


class SearchSource(Enum):
    """Search sources for interview data"""
    GLASSDOOR = "glassdoor"
    INDEED = "indeed"
    LEETCODE = "leetcode"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    GOOGLE = "google"
    REDDIT = "reddit"


class JobAnalyzer:
    """Analyzes job descriptions to extract key requirements"""

    def __init__(self):
        self.technical_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'go', 'rust', 'typescript', 'c#', 'swift',
                            'kotlin', 'scala', 'php', 'perl', 'r', 'matlab', 'julia'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'rails', 'laravel',
                           'fastapi', 'nextjs', 'nuxt', 'svelte', '.net', 'asp.net'],
            'databases': ['sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'dynamodb',
                          'firebase', 'oracle', 'sql server', 'mariadb', 'neo4j'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'kubernetes', 'docker', 'terraform', 'jenkins', 'circleci',
                      'github actions', 'cloudformation', 'ansible'],
            'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision',
                      'scikit-learn', 'keras', 'opencv', 'transformers', 'hugging face'],
            'data': ['etl', 'data pipeline', 'spark', 'hadoop', 'airflow', 'data warehouse', 'bi', 'tableau',
                     'power bi', 'looker', 'databricks', 'snowflake'],
            'mobile': ['ios', 'android', 'react native', 'flutter', 'swift', 'kotlin', 'xamarin'],
            'devops': ['ci/cd', 'devops', 'gitlab', 'bitbucket', 'jira', 'confluence', 'monitoring', 'logging',
                       'prometheus', 'grafana', 'elk stack'],
        }

    def analyze_job_description(self, job_description: str, company_info: str = "") -> Dict:
        """Extract key information from job description"""

        # Convert to lowercase for analysis
        jd_lower = job_description.lower()

        # Extract technical skills
        technical_skills = []
        for category, keywords in self.technical_keywords.items():
            for keyword in keywords:
                if keyword in jd_lower:
                    technical_skills.append(keyword)

        # Remove duplicates and sort by frequency
        technical_skills = list(set(technical_skills))

        # Extract years of experience
        experience_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience'
        experience_matches = re.findall(experience_pattern, jd_lower)
        years_experience = max(map(int, experience_matches)) if experience_matches else 0

        # Extract education requirements
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'computer science', 'engineering', 'mathematics',
                              'physics']
        education_requirements = [kw for kw in education_keywords if kw in jd_lower]

        # Determine role level
        if any(term in jd_lower for term in ['senior', 'lead', 'principal', 'staff', 'architect']):
            role_level = 'Senior'
        elif any(term in jd_lower for term in ['junior', 'entry', 'associate', 'intern']):
            role_level = 'Junior'
        else:
            role_level = 'Mid-level'

        # Extract soft skills
        soft_skills_keywords = [
            'communication', 'teamwork', 'leadership', 'problem-solving',
            'analytical', 'creative', 'organized', 'detail-oriented',
            'collaborative', 'innovative', 'strategic', 'mentoring'
        ]
        soft_skills = [skill for skill in soft_skills_keywords if skill in jd_lower]

        return {
            'technical_skills': technical_skills,
            'years_experience': years_experience,
            'education_requirements': education_requirements,
            'role_level': role_level,
            'soft_skills': soft_skills,
            'raw_description': job_description,
            'company_info': company_info
        }

@dataclass
class SearchResult:
    """Search result data structure"""
    source: SearchSource
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class InterviewQuestion:
    """Enhanced interview question with source data"""
    category: str
    question: str
    answer: str
    tips: List[str]
    difficulty: str
    keywords: List[str]
    source: Optional[str] = None
    real_examples: List[str] = field(default_factory=list)
    company_specific: bool = False
    ai_confidence: float = 0.0


class AdvancedWebSearcher:
    """Advanced web search with multiple sources"""

    def __init__(self):
        self.ddgs = DDGS()
        self.session = None
        self.cache = {}

    async def search_multiple_sources(self, query: str, sources: List[SearchSource]) -> List[SearchResult]:
        """Search across multiple sources concurrently"""
        results = []

        # Create async session
        async with aiohttp.ClientSession() as session:
            self.session = session

            # Create search tasks for each source
            tasks = []
            for source in sources:
                if source == SearchSource.GLASSDOOR:
                    tasks.append(self._search_glassdoor(query))
                elif source == SearchSource.INDEED:
                    tasks.append(self._search_indeed(query))
                elif source == SearchSource.LEETCODE:
                    tasks.append(self._search_leetcode(query))
                elif source == SearchSource.GITHUB:
                    tasks.append(self._search_github(query))
                elif source == SearchSource.REDDIT:
                    tasks.append(self._search_reddit(query))
                else:
                    tasks.append(self._search_google(query))

            # Execute all searches concurrently
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            for result_list in search_results:
                if isinstance(result_list, list):
                    results.extend(result_list)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:Config.MAX_SEARCH_RESULTS]

    async def _search_google(self, query: str) -> List[SearchResult]:
        """Search using DuckDuckGo (Google alternative)"""
        try:
            results = []
            search_results = self.ddgs.text(query, max_results=5)

            for item in search_results:
                results.append(SearchResult(
                    source=SearchSource.GOOGLE,
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('body', ''),
                    relevance_score=0.8
                ))

            return results
        except Exception as e:
            st.error(f"Google search error: {str(e)}")
            return []

    async def _search_glassdoor(self, query: str) -> List[SearchResult]:
        """Search Glassdoor for interview experiences"""
        try:
            # Glassdoor-specific search
            glassdoor_query = f"site:glassdoor.com {query} interview questions"
            results = []

            search_results = self.ddgs.text(glassdoor_query, max_results=3)

            for item in search_results:
                results.append(SearchResult(
                    source=SearchSource.GLASSDOOR,
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('body', ''),
                    metadata={'type': 'interview_experience'},
                    relevance_score=0.9
                ))

            return results
        except Exception as e:
            st.error(f"Glassdoor search error: {str(e)}")
            return []

    async def _search_indeed(self, query: str) -> List[SearchResult]:
        """Search Indeed for interview tips"""
        try:
            indeed_query = f"site:indeed.com {query} interview tips"
            results = []

            search_results = self.ddgs.text(indeed_query, max_results=3)

            for item in search_results:
                results.append(SearchResult(
                    source=SearchSource.INDEED,
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('body', ''),
                    metadata={'type': 'interview_guide'},
                    relevance_score=0.85
                ))

            return results
        except Exception as e:
            st.error(f"Indeed search error: {str(e)}")
            return []

    async def _search_leetcode(self, query: str) -> List[SearchResult]:
        """Search LeetCode for coding questions"""
        try:
            # Extract technical terms for LeetCode search
            tech_terms = re.findall(r'\b(?:algorithm|data structure|coding|programming|leetcode)\b', query.lower())
            if tech_terms:
                leetcode_query = f"site:leetcode.com {' '.join(tech_terms)} interview"
                results = []

                search_results = self.ddgs.text(leetcode_query, max_results=3)

                for item in search_results:
                    results.append(SearchResult(
                        source=SearchSource.LEETCODE,
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('body', ''),
                        metadata={'type': 'coding_problem'},
                        relevance_score=0.95
                    ))

                return results
        except Exception as e:
            st.error(f"LeetCode search error: {str(e)}")
        return []

    async def _search_github(self, query: str) -> List[SearchResult]:
        """Search GitHub for interview preparation repositories"""
        try:
            github_query = f"site:github.com {query} interview questions awesome"
            results = []

            search_results = self.ddgs.text(github_query, max_results=2)

            for item in search_results:
                results.append(SearchResult(
                    source=SearchSource.GITHUB,
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('body', ''),
                    metadata={'type': 'resource_collection'},
                    relevance_score=0.8
                ))

            return results
        except Exception as e:
            st.error(f"GitHub search error: {str(e)}")
            return []

    async def _search_reddit(self, query: str) -> List[SearchResult]:
        """Search Reddit for interview discussions"""
        try:
            reddit_query = f"site:reddit.com {query} interview experience"
            results = []

            search_results = self.ddgs.text(reddit_query, max_results=2)

            for item in search_results:
                results.append(SearchResult(
                    source=SearchSource.REDDIT,
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('body', ''),
                    metadata={'type': 'community_discussion'},
                    relevance_score=0.75
                ))

            return results
        except Exception as e:
            st.error(f"Reddit search error: {str(e)}")
            return []

    def extract_content_from_url(self, url: str) -> Optional[str]:
        """Extract detailed content from URL using Selenium"""
        try:
            # Setup Chrome options
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')

            # Initialize driver
            driver = webdriver.Chrome(
                ChromeDriverManager().install(),
                options=chrome_options
            )

            driver.get(url)

            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Extract text content
            content = driver.find_element(By.TAG_NAME, "body").text
            driver.quit()

            return content
        except Exception as e:
            st.warning(f"Could not extract content from {url}: {str(e)}")
            return None


class AdvancedAIGenerator:
    """Advanced AI model integration for question generation"""

    def __init__(self, model: AIModel = AIModel.CLAUDE_OPUS_4):
        self.model = model
        self.anthropic_client = None
        self.openai_client = None

        # Initialize clients based on available API keys
        if Config.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_questions_with_context(
            self,
            job_analysis: Dict,
            search_results: List[SearchResult],
            num_questions: int = 15
    ) -> List[InterviewQuestion]:
        """Generate questions using AI with web search context"""

        # Prepare context from search results
        context = self._prepare_context(search_results)

        # Select AI model and generate
        if self.model in [AIModel.CLAUDE_OPUS_4, AIModel.CLAUDE_SONNET_4] and self.anthropic_client:
            return await self._generate_with_claude(job_analysis, context, num_questions)
        elif self.model in [AIModel.GPT_4, AIModel.GPT_4_TURBO] and Config.OPENAI_API_KEY:
            return await self._generate_with_gpt4(job_analysis, context, num_questions)
        else:
            # Fallback to template-based generation
            return self._generate_template_questions(job_analysis, context)

    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context from search results"""
        context_parts = []

        for result in search_results[:5]:  # Top 5 results
            context_parts.append(f"""
Source: {result.source.value}
Title: {result.title}
Content: {result.snippet}
URL: {result.url}
---
""")

        return "\n".join(context_parts)

    async def _generate_with_claude(
            self,
            job_analysis: Dict,
            context: str,
            num_questions: int
    ) -> List[InterviewQuestion]:
        """Generate questions using Claude with better error handling"""

        prompt = f"""You are an expert technical interviewer with deep knowledge across all technical domains.

Job Analysis:
- Technical Skills: {', '.join(job_analysis['technical_skills'])}
- Role Level: {job_analysis['role_level']}
- Years Experience: {job_analysis['years_experience']}
- Company: {job_analysis.get('company_name', 'Not specified')}

Real Interview Data from Web:
{context}

Generate {num_questions} interview questions that:
1. Are highly specific to the technical skills required
2. Include real-world scenarios from the search results
3. Range from medium to hard difficulty
4. Cover technical, system design, and behavioral aspects
5. Include detailed answers with code examples where relevant

IMPORTANT: Return ONLY a valid JSON array, no other text. Format:
[
    {{
    "category": "Technical/System Design/Behavioral/Coding",
    "question": "The actual question",
    "answer": "Comprehensive answer with examples",
    "tips": ["tip1", "tip2", "tip3"],
    "difficulty": "Easy/Medium/Hard",
    "keywords": ["keyword1", "keyword2"],
    "real_examples": ["example1", "example2"],
    "company_specific": false
    }}
]"""

        try:
            message = self.anthropic_client.messages.create(
                model=self.model.value,
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Get response text
            response_text = message.content[0].text.strip()

            # Try to extract JSON from the response
            # Sometimes Claude adds explanation before/after JSON
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                response_text = json_match.group()

            # Parse JSON
            try:
                questions_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to clean up common issues
                response_text = response_text.replace('```json', '').replace('```', '')
                response_text = response_text.strip()
                questions_data = json.loads(response_text)

            # Convert to InterviewQuestion objects
            questions = []
            for q_data in questions_data:
                questions.append(InterviewQuestion(
                    category=q_data.get('category', 'Technical'),
                    question=q_data.get('question', ''),
                    answer=q_data.get('answer', ''),
                    tips=q_data.get('tips', []),
                    difficulty=q_data.get('difficulty', 'Medium'),
                    keywords=q_data.get('keywords', []),
                    real_examples=q_data.get('real_examples', []),
                    company_specific=q_data.get('company_specific', False),
                    ai_confidence=0.95,
                    source="Claude"
                ))

            return questions

        except json.JSONDecodeError as e:
            st.error(f"Claude JSON parsing error: {str(e)}")
            st.info("Falling back to template generation...")
            return self._generate_template_questions(job_analysis, context)
        except Exception as e:
            st.error(f"Claude generation error: {str(e)}")
            return self._generate_template_questions(job_analysis, context)

    async def _generate_with_gpt4(
            self,
            job_analysis: Dict,
            context: str,
            num_questions: int
    ) -> List[InterviewQuestion]:
        """Generate questions using GPT-4"""

        prompt = f"""As an expert technical interviewer, generate {num_questions} interview questions based on:

Job Requirements:
{json.dumps(job_analysis, indent=2)}

Real Interview Context:
{context}

Requirements:
1. Questions must be specific to the technical stack
2. Include questions actually asked in similar interviews
3. Provide comprehensive answers with examples
4. Cover all aspects: technical, system design, behavioral, coding

Format as JSON array with: category, question, answer, tips[], difficulty, keywords[], real_examples[], company_specific"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model.value,
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )

            # Parse response
            response_text = response.choices[0].message.content
            questions_data = json.loads(response_text)

            # Convert to InterviewQuestion objects
            questions = []
            for q_data in questions_data:
                questions.append(InterviewQuestion(
                    category=q_data['category'],
                    question=q_data['question'],
                    answer=q_data['answer'],
                    tips=q_data['tips'],
                    difficulty=q_data['difficulty'],
                    keywords=q_data['keywords'],
                    real_examples=q_data.get('real_examples', []),
                    company_specific=q_data.get('company_specific', False),
                    ai_confidence=0.93,
                    source="GPT-4"
                ))

            return questions

        except Exception as e:
            st.error(f"GPT-4 generation error: {str(e)}")
            return self._generate_template_questions(job_analysis, context)

    def _generate_template_questions(self, job_analysis: Dict, context: str) -> List[InterviewQuestion]:
        """Fallback template-based question generation"""
        questions = []

        # Extract insights from context
        context_lower = context.lower()

        # Technical questions for each skill
        for skill in job_analysis['technical_skills'][:5]:
            questions.append(InterviewQuestion(
                category="Technical",
                question=f"Explain your experience with {skill} and describe a challenging problem you solved using it.",
                answer=f"Structure your answer:\n"
                       f"1. Overview of your {skill} experience\n"
                       f"2. Specific project example\n"
                       f"3. Technical challenge faced\n"
                       f"4. Your solution approach\n"
                       f"5. Results and learnings",
                tips=[
                    f"Research latest {skill} best practices",
                    "Prepare 2-3 concrete examples",
                    "Be ready for follow-up technical questions",
                    "Mention relevant metrics or improvements"
                ],
                difficulty="Medium",
                keywords=[skill],
                real_examples=[],
                company_specific=False,
                ai_confidence=0.7,
                source="Template"
            ))

        return questions


def create_enhanced_pdf_report(
        questions: List[InterviewQuestion],
        job_info: Dict,
        search_results: List[SearchResult]
) -> bytes:
    """Create an enhanced PDF report with search insights"""

    report = f"""# Interview Preparation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Company:** {job_info.get('company_name', 'Not specified')}
**Role:** {job_info.get('role_title', 'Not specified')}
**Level:** {job_info.get('role_level', 'Not specified')}

## Key Skills
{', '.join(job_info.get('technical_skills', [])[:10])}

## Web Research Insights
Based on analysis of {len(search_results)} sources including Glassdoor, Indeed, and LeetCode:

"""

    # Add source summary
    source_counts = {}
    for result in search_results:
        source_counts[result.source.value] = source_counts.get(result.source.value, 0) + 1

    for source, count in source_counts.items():
        report += f"- {source.title()}: {count} relevant results\n"

    report += "\n---\n\n"

    # Group questions by category
    categories = {}
    for q in questions:
        if q.category not in categories:
            categories[q.category] = []
        categories[q.category].append(q)

    # Add questions by category
    for category, category_questions in categories.items():
        report += f"## {category} Questions\n\n"

        for i, q in enumerate(category_questions, 1):
            report += f"### {i}. {q.question}\n\n"
            report += f"**Difficulty:** {q.difficulty} | **Source:** {q.source}\n"
            report += f"**Confidence:** {q.ai_confidence:.0%}\n\n"

            report += f"**Answer Framework:**\n{q.answer}\n\n"

            if q.real_examples:
                report += f"**Real Interview Examples:**\n"
                for example in q.real_examples:
                    report += f"- {example}\n"
                report += "\n"

            report += f"**Preparation Tips:**\n"
            for tip in q.tips:
                report += f"- {tip}\n"

            report += f"\n**Keywords:** {', '.join(q.keywords)}\n"
            report += "\n---\n\n"

    # Add resources section
    report += "## Additional Resources\n\n"
    for result in search_results[:10]:
        report += f"- [{result.title}]({result.url})\n"

    return report.encode('utf-8')


def main():
    """Enhanced Streamlit application"""
    st.set_page_config(
        page_title="AI Interview Prep Assistant",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Advanced Interview Preparation Assistant")
    st.markdown("Powered by Claude Opus 4, GPT-4, and real-time web search")

    # Initialize cache
    #cache = InterviewCache()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # AI Model Selection
        st.subheader("AI Model")
        available_models = []

        if Config.ANTHROPIC_API_KEY:
            available_models.extend([AIModel.CLAUDE_OPUS_4, AIModel.CLAUDE_SONNET_4])
        if Config.OPENAI_API_KEY:
            available_models.extend([AIModel.GPT_4, AIModel.GPT_4_TURBO])

        if available_models:
            selected_model = st.selectbox(
                "Select AI Model",
                available_models,
                format_func=lambda x: x.value
            )
        else:
            st.warning("No AI API keys configured")
            selected_model = None

        # API Key inputs
        with st.expander("API Keys"):
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=Config.ANTHROPIC_API_KEY[:10] + "..." if Config.ANTHROPIC_API_KEY else "",
                type="password"
            )
            if anthropic_key and not anthropic_key.endswith("..."):
                Config.ANTHROPIC_API_KEY = anthropic_key

            openai_key = st.text_input(
                "OpenAI API Key",
                value=Config.OPENAI_API_KEY[:10] + "..." if Config.OPENAI_API_KEY else "",
                type="password"
            )
            if openai_key and not openai_key.endswith("..."):
                Config.OPENAI_API_KEY = openai_key

        # Search Sources
        st.subheader("Search Sources")
        search_sources = st.multiselect(
            "Select sources to search",
            [source for source in SearchSource],
            default=[SearchSource.GLASSDOOR, SearchSource.LEETCODE, SearchSource.GOOGLE],
            format_func=lambda x: x.value.title()
        )

        # Advanced Options
        st.subheader("Advanced Options")
        num_questions = st.slider("Number of questions", 5, 30, 15)
        include_coding = st.checkbox("Include coding problems", value=True)
        include_system_design = st.checkbox("Include system design", value=True)
        #use_cache = st.checkbox("Use cached results", value=True)

        st.divider()

        # Info
        st.info("""
        **Features:**
        - Real-time web search
        - AI-powered analysis
        - Company-specific insights
        - Actual interview questions
        - Comprehensive answers
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Job Analysis",
        "🔍 Web Research",
        "❓ Interview Questions",
        "📚 Resources"
    ])

    with tab1:
        st.header("Job Description Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            job_description = st.text_area(
                "Paste the job description",
                height=400,
                placeholder="Copy and paste the complete job description here..."
            )

        with col2:
            company_name = st.text_input("Company Name", placeholder="e.g., Google")
            role_title = st.text_input("Role Title", placeholder="e.g., Senior Software Engineer")

            st.divider()

            # Quick templates
            st.subheader("Quick Templates")
            if st.button("Load SWE Example"):
                job_description = """
                Senior Software Engineer - Full Stack

                We are looking for a Senior Software Engineer with:
                - 5+ years of experience with Python, React, and AWS
                - Strong background in distributed systems
                - Experience with microservices architecture
                - Knowledge of ML/AI is a plus
                - Excellent problem-solving skills
                """
                company_name = "TechCorp"
                role_title = "Senior Software Engineer"

        if st.button("🔍 Analyze & Search", type="primary", key="analyze"):
            if job_description:
                with st.spinner("Analyzing job description..."):
                    # Basic analysis
                    analyzer = JobAnalyzer()
                    job_analysis = analyzer.analyze_job_description(job_description)
                    job_analysis['company_name'] = company_name
                    job_analysis['role_title'] = role_title

                    # Store in session
                    st.session_state['job_analysis'] = job_analysis

                    # Display results
                    st.success("✅ Analysis complete!")

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Role Level", job_analysis['role_level'])
                    with col2:
                        st.metric("Experience", f"{job_analysis['years_experience']}+ years")
                    with col3:
                        st.metric("Tech Skills", len(job_analysis['technical_skills']))
                    with col4:
                        st.metric("Soft Skills", len(job_analysis['soft_skills']))

                    # Skills display
                    if job_analysis['technical_skills']:
                        st.subheader("🔧 Technical Skills Identified")
                        skills_cols = st.columns(5)
                        for i, skill in enumerate(job_analysis['technical_skills'][:15]):
                            skills_cols[i % 5].badge(skill, icon="⚡")

                # Trigger web search
                with st.spinner("🌐 Searching web for interview data..."):
                    searcher = AdvancedWebSearcher()

                    # Create search queries
                    search_queries = [
                        f"{company_name} {role_title} interview questions",
                        f"{company_name} interview experience {datetime.now().year}",
                        f"{role_title} technical interview questions"
                    ]

                    # Add skill-specific searches
                    for skill in job_analysis['technical_skills'][:3]:
                        search_queries.append(f"{skill} interview questions {role_title}")

                    # Perform searches
                    all_results = []
                    for query in search_queries:
                        # Run async search
                        results = asyncio.run(
                            searcher.search_multiple_sources(query, search_sources)
                        )
                        all_results.extend(results)

                    # Store search results
                    st.session_state['search_results'] = all_results
                    st.success(f"✅ Found {len(all_results)} relevant sources!")
            else:
                st.warning("Please provide a job description")

    with tab2:
        st.header("🔍 Web Research Results")

        if 'search_results' in st.session_state:
            results = st.session_state['search_results']

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sources", len(results))
            with col2:
                unique_sources = len(set(r.source for r in results))
                st.metric("Unique Platforms", unique_sources)
            with col3:
                glassdoor_count = sum(1 for r in results if r.source == SearchSource.GLASSDOOR)
                st.metric("Glassdoor Results", glassdoor_count)

            # Filter by source
            selected_source = st.selectbox(
                "Filter by source",
                ["All"] + [s.value.title() for s in SearchSource]
            )

            # Display results
            for idx, result in enumerate(results):
                if selected_source == "All" or result.source.value == selected_source.lower():
                    with st.expander(f"{result.source.value.title()}: {result.title[:60]}..."):
                        st.markdown(f"**URL:** [{result.url}]({result.url})")
                        st.markdown(f"**Relevance:** {result.relevance_score:.0%}")
                        st.markdown(f"**Summary:** {result.snippet}")

                        if st.button(f"Extract Full Content", key=f"extract_{idx}"):
                            with st.spinner("Extracting content..."):
                                searcher = AdvancedWebSearcher()
                                content = searcher.extract_content_from_url(result.url)
                                if content:
                                    st.text_area("Full Content", content[:2000], height=300)
        else:
            st.info("👈 Analyze a job description first to see search results")

    with tab3:
        st.header("❓ AI-Generated Interview Questions")

        if 'job_analysis' in st.session_state and 'search_results' in st.session_state:
            if st.button("🤖 Generate Interview Questions", type="primary", key="generate"):
                if selected_model:
                    with st.spinner(f"Generating questions with {selected_model.value}..."):
                        # Initialize AI generator
                        ai_generator = AdvancedAIGenerator(selected_model)

                        # Generate questions
                        questions = asyncio.run(
                            ai_generator.generate_questions_with_context(
                                st.session_state['job_analysis'],
                                st.session_state['search_results'],
                                num_questions
                            )
                        )

                        # Store questions
                        st.session_state['questions'] = questions
                        st.success(f"✅ Generated {len(questions)} questions!")
                else:
                    st.error("Please configure at least one AI API key")

            # Display questions
            if 'questions' in st.session_state:
                questions = st.session_state['questions']

                # Category filter
                categories = list(set(q.category for q in questions))
                selected_category = st.selectbox(
                    "Filter by category",
                    ["All"] + categories
                )

                # Difficulty filter
                difficulties = list(set(q.difficulty for q in questions))
                selected_difficulty = st.selectbox(
                    "Filter by difficulty",
                    ["All"] + difficulties
                )

                # Display questions
                displayed_count = 0
                for i, question in enumerate(questions, 1):
                    if (selected_category == "All" or question.category == selected_category) and \
                            (selected_difficulty == "All" or question.difficulty == selected_difficulty):

                        displayed_count += 1
                        with st.expander(
                                f"Q{displayed_count}: {question.question}",
                                expanded=displayed_count <= 3
                        ):
                            # Question metadata
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f"**Category:** {question.category}")
                            with col2:
                                st.markdown(f"**Difficulty:** {question.difficulty}")
                            with col3:
                                st.markdown(f"**Source:** {question.source}")
                            with col4:
                                st.markdown(f"**Confidence:** {question.ai_confidence:.0%}")

                            st.divider()

                            # Answer
                            st.markdown("### 📝 Comprehensive Answer")
                            st.markdown(question.answer)

                            # Real examples if available
                            if question.real_examples:
                                st.markdown("### 🌟 Real Interview Examples")
                                for example in question.real_examples:
                                    st.markdown(f"- {example}")

                            # Tips
                            st.markdown("### 💡 Preparation Tips")
                            for tip in question.tips:
                                st.markdown(f"- {tip}")

                            # Keywords
                            st.markdown("### 🏷️ Key Terms")
                            for keyword in question.keywords:
                                st.badge(keyword)

                # Export options
                st.divider()

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📄 Generate PDF Report"):
                        pdf_content = create_enhanced_pdf_report(
                            questions,
                            st.session_state['job_analysis'],
                            st.session_state['search_results']
                        )
                        st.download_button(
                            "Download PDF",
                            data=pdf_content,
                            file_name=f"interview_prep_{company_name}_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )

                with col2:
                    if st.button("📊 Export to JSON"):
                        json_data = json.dumps([
                            {
                                'category': q.category,
                                'question': q.question,
                                'answer': q.answer,
                                'tips': q.tips,
                                'difficulty': q.difficulty,
                                'keywords': q.keywords,
                                'source': q.source
                            }
                            for q in questions
                        ], indent=2)
                        st.download_button(
                            "Download JSON",
                            data=json_data,
                            file_name=f"questions_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )

                with col3:
                    if st.button("🔄 Regenerate Questions"):
                        st.session_state.pop('questions', None)
                        st.experimental_rerun()
        else:
            st.info("👈 Complete job analysis and web search first")

    with tab4:
        st.header("📚 Additional Resources & Practice")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Interview Practice Platforms")
            platforms = {
                "Pramp": "https://pramp.com - Free peer mock interviews",
                "Interviewing.io": "https://interviewing.io - Anonymous technical interviews",
                "LeetCode": "https://leetcode.com - Coding practice",
                "System Design Primer": "https://github.com/donnemartin/system-design-primer",
                "Glassdoor": "https://glassdoor.com - Company reviews & questions"
            }

            for name, info in platforms.items():
                st.markdown(f"**{name}**: {info}")

        with col2:
            st.subheader("📖 Recommended Books")
            books = [
                "Cracking the Coding Interview - Gayle McDowell",
                "System Design Interview - Alex Xu",
                "Designing Data-Intensive Applications - Martin Kleppmann",
                "The Pragmatic Programmer - David Thomas",
                "Clean Code - Robert Martin"
            ]

            for book in books:
                st.markdown(f"- {book}")

        # Interview checklist
        st.divider()
        st.subheader("✅ Pre-Interview Checklist")

        checklist_items = [
            "Research company culture and recent news",
            "Practice answers to generated questions",
            "Prepare questions to ask interviewers",
            "Review resume and prepare to discuss each point",
            "Set up technical environment (IDE, camera, mic)",
            "Practice whiteboarding or screen sharing",
            "Prepare STAR examples for behavioral questions",
            "Research your interviewers on LinkedIn",
            "Plan your schedule and logistics",
            "Get good rest the night before"
        ]

        for item in checklist_items:
            st.checkbox(item, key=f"checklist_{item}")


if __name__ == "__main__":
    main()