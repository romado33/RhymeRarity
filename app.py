# enhanced_rhyme_checker.py
"""
Enhanced Rhyme Rarity Checker with improved features, error handling, and user experience
"""

import streamlit as st
import openai
import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load environment variables
load_dotenv()

# === Configuration ===
@dataclass
class RhymeConfig:
    """Configuration settings for the rhyme checker"""
    max_word_length: int = 50
    max_history_items: int = 50
    openai_model: str = "gpt-4"
    temperature: float = 0.3  # Lower for more consistent results
    max_tokens: int = 500
    cache_ttl_hours: int = 24

CONFIG = RhymeConfig()

# === Data Models ===
@dataclass
class RhymeAnalysis:
    """Data model for rhyme analysis results"""
    word1: str
    word2: str
    rarity_score: int
    rhyme_type: str
    explanation: str
    examples: str
    phonetic_similarity: str
    timestamp: datetime
    response_hash: str
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: dict):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

# === App Configuration ===
st.set_page_config(
    page_title="Rhyme Rarity Checker",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Session State Management ===
def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        'rhyme_history': [],
        'analysis_cache': {},
        'openai_available': None,
        'user_preferences': {
            'show_phonetics': True,
            'include_examples': True,
            'detailed_analysis': False
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === OpenAI Setup ===
def setup_openai() -> bool:
    """Setup and validate OpenAI API"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("🔑 OpenAI API key not found. Please set OPENAI_API_KEY in your environment or Streamlit secrets.")
        return False
    
    try:
        openai.api_key = api_key
        # Test the API with a minimal request
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        st.error("🚫 Invalid OpenAI API key. Please check your credentials.")
        return False
    except openai.error.RateLimitError:
        st.warning("⏰ Rate limit reached. Please try again later.")
        return False
    except Exception as e:
        st.error(f"❌ OpenAI setup failed: {str(e)}")
        return False

# === Input Validation ===
def validate_word(word: str) -> Tuple[bool, str]:
    """Validate input word"""
    if not word.strip():
        return False, "Word cannot be empty"
    
    if len(word) > CONFIG.max_word_length:
        return False, f"Word too long (max {CONFIG.max_word_length} characters)"
    
    # Check for valid characters (letters, apostrophes, hyphens)
    if not re.match(r"^[a-zA-Z\-']+$", word.strip()):
        return False, "Word contains invalid characters"
    
    return True, ""

def validate_word_pair(word1: str, word2: str) -> Tuple[bool, str]:
    """Validate word pair"""
    # Individual word validation
    is_valid1, msg1 = validate_word(word1)
    if not is_valid1:
        return False, f"First word: {msg1}"
    
    is_valid2, msg2 = validate_word(word2)
    if not is_valid2:
        return False, f"Second word: {msg2}"
    
    # Check if words are the same
    if word1.lower().strip() == word2.lower().strip():
        return False, "Please enter two different words"
    
    return True, ""

# === Caching System ===
def get_cache_key(word1: str, word2: str, preferences: dict) -> str:
    """Generate cache key for word pair and preferences"""
    # Normalize words (lowercase, stripped)
    normalized = f"{word1.lower().strip()}_{word2.lower().strip()}"
    prefs_str = json.dumps(preferences, sort_keys=True)
    combined = f"{normalized}_{prefs_str}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_analysis(cache_key: str) -> Optional[RhymeAnalysis]:
    """Retrieve cached analysis if available and not expired"""
    cache = st.session_state.analysis_cache
    
    if cache_key in cache:
        cached_data = cache[cache_key]
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        
        # Check if cache is still valid (within TTL)
        hours_since_cached = (datetime.now() - cached_time).total_seconds() / 3600
        if hours_since_cached < CONFIG.cache_ttl_hours:
            return RhymeAnalysis.from_dict(cached_data)
    
    return None

def cache_analysis(cache_key: str, analysis: RhymeAnalysis):
    """Cache the analysis result"""
    st.session_state.analysis_cache[cache_key] = analysis.to_dict()
    
    # Limit cache size to prevent memory issues
    if len(st.session_state.analysis_cache) > 100:
        # Remove oldest entries
        items = list(st.session_state.analysis_cache.items())
        items.sort(key=lambda x: x[1]['timestamp'])
        for key, _ in items[:20]:
            del st.session_state.analysis_cache[key]

# === Enhanced Prompt Building ===
def build_analysis_prompt(word1: str, word2: str, preferences: dict) -> str:
    """Build enhanced prompt for rhyme analysis"""
    
    base_prompt = f"""
You are an expert linguist and poetry analyst specializing in rhyme patterns and phonetics.

TASK: Analyze the rhyme relationship between "{word1}" and "{word2}".

ANALYSIS REQUIREMENTS:
1. Rarity Score (0-100): Rate how common/rare this rhyme pairing is
   - 0-20: Extremely common (like "cat/hat", "love/dove")
   - 21-40: Common but useful
   - 41-60: Moderately rare, interesting
   - 61-80: Rare and creative
   - 81-100: Extremely rare or forced

2. Rhyme Type: Classify as one of:
   - Perfect: Identical sounds from vowel onward (cat/hat)
   - Near/Slant: Similar but not identical sounds (heart/part)
   - Eye: Visual similarity but different sounds (love/move)
   - Forced: Awkward or unnatural pairing
   - None: No meaningful rhyme relationship

3. Explanation: Brief analysis of the phonetic relationship and usage frequency

4. Examples: Real examples from literature, song lyrics, or poetry (if any exist)
"""

    if preferences.get('show_phonetics', True):
        base_prompt += "\n5. Phonetic Analysis: Describe the sound patterns and phonetic elements"

    if preferences.get('detailed_analysis', False):
        base_prompt += "\n6. Literary Analysis: Discuss effectiveness and emotional impact of this rhyme pair"

    base_prompt += """

FORMAT YOUR RESPONSE EXACTLY AS:
Rarity Score: [number]
Rhyme Type: [type]
Explanation: [2-3 sentences]
Examples: [specific examples or "None found"]"""

    if preferences.get('show_phonetics', True):
        base_prompt += "\nPhonetic Analysis: [analysis]"

    if preferences.get('detailed_analysis', False):
        base_prompt += "\nLiterary Analysis: [analysis]"

    return base_prompt

# === AI Analysis ===
def get_rhyme_analysis(word1: str, word2: str, preferences: dict) -> RhymeAnalysis:
    """Get rhyme analysis from OpenAI with error handling"""
    
    prompt = build_analysis_prompt(word1, word2, preferences)
    
    try:
        response = openai.ChatCompletion.create(
            model=CONFIG.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise rhyme and phonetics expert. Always follow the exact format requested and provide accurate, consistent scoring."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG.temperature,
            max_tokens=CONFIG.max_tokens,
            timeout=30
        )
        
        content = response.choices[0].message.content
        response_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Parse the response
        analysis = parse_analysis_response(word1, word2, content, response_hash)
        return analysis
        
    except openai.error.RateLimitError:
        st.error("⏰ Rate limit exceeded. Please wait a minute and try again.")
        raise
    except openai.error.InvalidRequestError as e:
        st.error(f"❌ Invalid request: {str(e)}")
        raise
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        raise

# === Response Parsing ===
def parse_analysis_response(word1: str, word2: str, content: str, response_hash: str) -> RhymeAnalysis:
    """Parse the AI response into structured data"""
    
    # Default values
    rarity_score = 0
    rhyme_type = "Unknown"
    explanation = "Unable to parse response"
    examples = "None"
    phonetic_similarity = "Not analyzed"
    
    try:
        # Extract rarity score
        score_match = re.search(r'Rarity Score:\s*(\d+)', content, re.IGNORECASE)
        if score_match:
            rarity_score = min(100, max(0, int(score_match.group(1))))
        
        # Extract rhyme type
        type_match = re.search(r'Rhyme Type:\s*([^\n]+)', content, re.IGNORECASE)
        if type_match:
            rhyme_type = type_match.group(1).strip()
        
        # Extract explanation
        exp_match = re.search(r'Explanation:\s*([^\n]+(?:\n[^\n\w:]*[^\n]*)*)', content, re.IGNORECASE)
        if exp_match:
            explanation = exp_match.group(1).strip()
        
        # Extract examples
        ex_match = re.search(r'Examples:\s*([^\n]+(?:\n[^\n\w:]*[^\n]*)*)', content, re.IGNORECASE)
        if ex_match:
            examples = ex_match.group(1).strip()
        
        # Extract phonetic analysis if present
        phon_match = re.search(r'Phonetic Analysis:\s*([^\n]+(?:\n[^\n\w:]*[^\n]*)*)', content, re.IGNORECASE)
        if phon_match:
            phonetic_similarity = phon_match.group(1).strip()
            
    except Exception as e:
        st.warning(f"⚠️ Response parsing incomplete: {str(e)}")
    
    return RhymeAnalysis(
        word1=word1,
        word2=word2,
        rarity_score=rarity_score,
        rhyme_type=rhyme_type,
        explanation=explanation,
        examples=examples,
        phonetic_similarity=phonetic_similarity,
        timestamp=datetime.now(),
        response_hash=response_hash
    )

# === Visualization ===
def create_rarity_gauge(score: int) -> go.Figure:
    """Create a gauge chart for rarity score"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Rarity Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 40], 'color': "gray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_history_chart(history: List[RhymeAnalysis]) -> go.Figure:
    """Create a chart showing rarity scores over time"""
    
    if not history:
        return None
    
    df = pd.DataFrame([
        {
            'pair': f"{analysis.word1}/{analysis.word2}",
            'score': analysis.rarity_score,
            'type': analysis.rhyme_type,
            'timestamp': analysis.timestamp
        }
        for analysis in history[-20:]  # Last 20 analyses
    ])
    
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='score',
        color='type',
        hover_data=['pair'],
        title="Rarity Score History",
        labels={'score': 'Rarity Score', 'timestamp': 'Time'}
    )
    
    fig.update_layout(height=400)
    return fig

# === Main UI Functions ===
def render_sidebar():
    """Render sidebar with preferences and history"""
    
    with st.sidebar:
        st.header("⚙️ Preferences")
        
        # User preferences
        prefs = st.session_state.user_preferences
        
        prefs['show_phonetics'] = st.checkbox(
            "🔊 Show phonetic analysis", 
            value=prefs['show_phonetics']
        )
        
        prefs['include_examples'] = st.checkbox(
            "📚 Include literary examples", 
            value=prefs['include_examples']
        )
        
        prefs['detailed_analysis'] = st.checkbox(
            "🔍 Detailed literary analysis", 
            value=prefs['detailed_analysis']
        )
        
        st.session_state.user_preferences = prefs
        
        # Statistics
        if st.session_state.rhyme_history:
            st.header("📊 Statistics")
            
            history = st.session_state.rhyme_history
            avg_score = sum(analysis.rarity_score for analysis in history) / len(history)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Checks", len(history))
            with col2:
                st.metric("Avg Rarity", f"{avg_score:.1f}")
            
            # Most common rhyme type
            rhyme_types = [analysis.rhyme_type for analysis in history]
            if rhyme_types:
                most_common = max(set(rhyme_types), key=rhyme_types.count)
                st.metric("Most Common Type", most_common)

def render_main_interface():
    """Render the main application interface"""
    
    # Header
    st.title("🎤 Enhanced Rhyme Rarity Checker")
    st.markdown("""
    **Discover how common or rare your rhyming word pairs are!**  
    Get detailed phonetic analysis, rarity scoring, and literary examples.
    """)
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        word1 = st.text_input(
            "First word:",
            placeholder="e.g., 'mountain'",
            help="Enter any English word"
        )
    
    with col2:
        word2 = st.text_input(
            "Second word:",
            placeholder="e.g., 'fountain'",
            help="Enter a word to rhyme with the first"
        )
    
    # Analysis button
    if st.button("🔍 Analyze Rhyme Rarity", type="primary", disabled=not (word1 and word2)):
        analyze_rhyme_pair(word1.strip(), word2.strip())

def analyze_rhyme_pair(word1: str, word2: str):
    """Main analysis function"""
    
    # Validate inputs
    is_valid, error_msg = validate_word_pair(word1, word2)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return
    
    # Check cache first
    cache_key = get_cache_key(word1, word2, st.session_state.user_preferences)
    cached_analysis = get_cached_analysis(cache_key)
    
    if cached_analysis:
        st.info("⚡ Using cached result for faster response!")
        display_analysis_results(cached_analysis)
        return
    
    # Perform new analysis
    with st.spinner("🤔 Analyzing rhyme rarity... This may take a moment."):
        try:
            analysis = get_rhyme_analysis(word1, word2, st.session_state.user_preferences)
            
            # Cache and store results
            cache_analysis(cache_key, analysis)
            st.session_state.rhyme_history.append(analysis)
            
            # Limit history size
            if len(st.session_state.rhyme_history) > CONFIG.max_history_items:
                st.session_state.rhyme_history.pop(0)
            
            display_analysis_results(analysis)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

def display_analysis_results(analysis: RhymeAnalysis):
    """Display the analysis results with enhanced formatting"""
    
    st.markdown("---")
    st.subheader(f"🎯 Analysis: '{analysis.word1}' & '{analysis.word2}'")
    
    # Main results in columns
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Rarity gauge
        gauge_fig = create_rarity_gauge(analysis.rarity_score)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        st.metric("Rarity Score", f"{analysis.rarity_score}/100")
        
        # Rarity interpretation
        if analysis.rarity_score <= 20:
            rarity_desc = "Very Common"
            color = "🟢"
        elif analysis.rarity_score <= 40:
            rarity_desc = "Common"
            color = "🟡"
        elif analysis.rarity_score <= 60:
            rarity_desc = "Moderate"
            color = "🟠"
        elif analysis.rarity_score <= 80:
            rarity_desc = "Rare"
            color = "🔴"
        else:
            rarity_desc = "Very Rare"
            color = "🟣"
        
        st.write(f"{color} **{rarity_desc}**")
    
    with col3:
        st.metric("Rhyme Type", analysis.rhyme_type)
    
    # Detailed analysis
    st.markdown("### 📝 Detailed Analysis")
    
    with st.container():
        st.markdown("**Explanation:**")
        st.write(analysis.explanation)
        
        if analysis.examples and analysis.examples != "None":
            st.markdown("**Examples:**")
            st.write(analysis.examples)
        
        if st.session_state.user_preferences.get('show_phonetics') and analysis.phonetic_similarity != "Not analyzed":
            st.markdown("**Phonetic Analysis:**")
            st.write(analysis.phonetic_similarity)
    
    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        json_data = json.dumps(analysis.to_dict(), indent=2, default=str)
        st.download_button(
            "📊 Download JSON",
            json_data,
            file_name=f"rhyme_analysis_{analysis.word1}_{analysis.word2}.json",
            mime="application/json"
        )
    
    with col2:
        # Text report
        report = f"""
Rhyme Analysis Report
====================
Words: {analysis.word1} & {analysis.word2}
Rarity Score: {analysis.rarity_score}/100
Rhyme Type: {analysis.rhyme_type}
Date: {analysis.timestamp.strftime('%Y-%m-%d %H:%M')}

Explanation:
{analysis.explanation}

Examples:
{analysis.examples}

Phonetic Analysis:
{analysis.phonetic_similarity}
        """.strip()
        
        st.download_button(
            "📄 Download Report",
            report,
            file_name=f"rhyme_report_{analysis.word1}_{analysis.word2}.txt",
            mime="text/plain"
        )

def render_history_section():
    """Render the history section"""
    
    if not st.session_state.rhyme_history:
        return
    
    if st.checkbox("📚 Show Analysis History"):
        st.markdown("### 📈 Analysis History")
        
        # History chart
        history_fig = create_history_chart(st.session_state.rhyme_history)
        if history_fig:
            st.plotly_chart(history_fig, use_container_width=True)
        
        # Recent analyses table
        if len(st.session_state.rhyme_history) > 0:
            st.markdown("#### Recent Analyses")
            
            recent_data = []
            for analysis in reversed(st.session_state.rhyme_history[-10:]):
                recent_data.append({
                    "Word Pair": f"{analysis.word1} & {analysis.word2}",
                    "Rarity": analysis.rarity_score,
                    "Type": analysis.rhyme_type,
                    "Date": analysis.timestamp.strftime("%m/%d %H:%M")
                })
            
            df = pd.DataFrame(recent_data)
            st.dataframe(df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.rhyme_history = []
            st.session_state.analysis_cache = {}
            st.rerun()

# === Main Application ===
def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Setup OpenAI
    if st.session_state.openai_available is None:
        st.session_state.openai_available = setup_openai()
    
    if not st.session_state.openai_available:
        st.stop()
    
    # Render UI components
    render_sidebar()
    render_main_interface()
    render_history_section()

if __name__ == "__main__":
    main()