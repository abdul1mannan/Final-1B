"""
Persona Manager - Advanced persona definitions and job-to-be-done matching
Enhanced with semantic analysis and adaptive learning
"""

import logging
import yaml
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path


class PersonaType(Enum):
    """Dynamic persona types - can be extended at runtime"""
    DYNAMIC = "dynamic"  # Default for any persona not predefined

class JobCategory(Enum):
    """Dynamic job categories - can be extended at runtime"""
    DYNAMIC = "dynamic"  # Default for any job not predefined

class PersonaCluster(Enum):
    """High-level persona clusters for pattern matching"""
    ANALYTICAL = "analytical"
    BUSINESS = "business"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    SERVICE = "service"
    CREATIVE = "creative"
    DYNAMIC = "dynamic"  # For personas that don't fit predefined clusters

@dataclass
class PersonaProfile:
    """Dynamic persona profile that adapts to any persona type"""
    persona_id: str
    original_persona: str  # Store the original persona string
    persona_type: PersonaType = PersonaType.DYNAMIC
    persona_cluster: PersonaCluster = PersonaCluster.DYNAMIC
    expertise_areas: List[str] = field(default_factory=list)
    preferred_sections: List[str] = field(default_factory=list)
    detail_level: str = "detailed"  # "summary", "detailed", "comprehensive"
    max_reading_time: int = 180  # minutes
    job_categories: List[str] = field(default_factory=list)  # Changed to strings for flexibility
    keywords: List[str] = field(default_factory=list)
    exclusion_keywords: List[str] = field(default_factory=list)
    
    # Enhanced fields
    confidence_threshold: float = 0.3
    section_priorities: Dict[str, float] = field(default_factory=dict)
    content_preferences: Dict[str, float] = field(default_factory=dict)
    time_sensitivity: str = "medium"  # "low", "medium", "high"
    complexity_preference: str = "balanced"  # "simple", "balanced", "complex"
    
    # Dynamic fields
    semantic_keywords: Set[str] = field(default_factory=set)
    domain_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if not self.section_priorities:
            self.section_priorities = self._compute_default_priorities()
        if not self.semantic_keywords:
            self.semantic_keywords = set(self.keywords + self.expertise_areas)
    
    def _compute_default_priorities(self) -> Dict[str, float]:
        """Compute default section priorities based on persona characteristics"""
        priorities = {}
        
        # Default balanced priorities
        default_sections = [
            "summary", "overview", "introduction", "main", "important", 
            "key", "conclusion", "recommendations", "results"
        ]
        
        for section in default_sections:
            priorities[section] = 0.7
        
        # Adjust based on persona cluster
        if self.persona_cluster == PersonaCluster.ANALYTICAL:
            priorities.update({
                "methodology": 0.9, "results": 0.9, "analysis": 0.8,
                "data": 0.8, "findings": 0.8, "conclusion": 0.7
            })
        elif self.persona_cluster == PersonaCluster.BUSINESS:
            priorities.update({
                "executive_summary": 1.0, "recommendations": 0.9,
                "financial": 0.8, "strategy": 0.8, "market": 0.7
            })
        elif self.persona_cluster == PersonaCluster.TECHNICAL:
            priorities.update({
                "implementation": 0.9, "architecture": 0.8,
                "specifications": 0.8, "performance": 0.7
            })
        elif self.persona_cluster == PersonaCluster.EDUCATIONAL:
            priorities.update({
                "introduction": 0.8, "concepts": 0.9, "examples": 0.8,
                "practice": 0.7, "summary": 0.8
            })
        elif self.persona_cluster == PersonaCluster.SERVICE:
            priorities.update({
                "tips": 0.9, "recommendations": 0.9, "guide": 0.8,
                "how_to": 0.8, "steps": 0.8, "practical": 0.7
            })
        
        return priorities

class PersonaManager:
    """Dynamic persona management with intelligent pattern recognition"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize dynamic components
        self.domain_patterns = self._load_domain_patterns()
        self.job_patterns = self._load_job_patterns()
        self.section_patterns = self._load_section_patterns()
        
        # Caching for performance
        self._persona_cache = {}
        self._job_analysis_cache = {}
        
        self.logger.info(f"PersonaManager initialized with dynamic pattern recognition")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "configs" / "model_config_1b.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('persona_intelligence', {})
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'confidence_threshold': 0.3,
            'max_keywords': 20,
            'enable_semantic_matching': True,
            'enable_adaptive_learning': False
        }
    
    def _load_domain_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load domain patterns for intelligent persona clustering"""
        return {
            PersonaCluster.ANALYTICAL.value: {
                'keywords': ['research', 'study', 'analysis', 'investigate', 'examine', 'data', 'findings', 'methodology', 'statistics', 'evidence'],
                'indicators': ['researcher', 'analyst', 'scientist', 'academic', 'scholar'],
                'sections': ['methodology', 'results', 'analysis', 'findings', 'data', 'research', 'study'],
                'complexity': 'complex',
                'detail_level': 'comprehensive'
            },
            PersonaCluster.BUSINESS.value: {
                'keywords': ['business', 'revenue', 'profit', 'market', 'strategy', 'financial', 'investment', 'growth', 'performance', 'roi'],
                'indicators': ['manager', 'executive', 'analyst', 'consultant', 'entrepreneur', 'director', 'ceo'],
                'sections': ['executive_summary', 'financial', 'market', 'strategy', 'performance', 'recommendations'],
                'complexity': 'balanced',
                'detail_level': 'detailed'
            },
            PersonaCluster.TECHNICAL.value: {
                'keywords': ['technical', 'system', 'architecture', 'implementation', 'development', 'engineering', 'software', 'hardware', 'design'],
                'indicators': ['engineer', 'developer', 'architect', 'technical', 'programmer', 'specialist'],
                'sections': ['architecture', 'implementation', 'technical', 'system', 'design', 'specifications'],
                'complexity': 'complex',
                'detail_level': 'comprehensive'
            },
            PersonaCluster.EDUCATIONAL.value: {
                'keywords': ['learn', 'teach', 'education', 'training', 'curriculum', 'knowledge', 'skill', 'concept', 'understanding'],
                'indicators': ['student', 'teacher', 'educator', 'instructor', 'trainer', 'learner'],
                'sections': ['introduction', 'concepts', 'examples', 'practice', 'summary', 'learning'],
                'complexity': 'simple',
                'detail_level': 'summary'
            },
            PersonaCluster.SERVICE.value: {
                'keywords': ['service', 'help', 'assist', 'support', 'guide', 'plan', 'organize', 'manage', 'coordinate'],
                'indicators': ['planner', 'coordinator', 'manager', 'specialist', 'advisor', 'consultant'],
                'sections': ['guide', 'tips', 'recommendations', 'steps', 'how_to', 'practical'],
                'complexity': 'balanced',
                'detail_level': 'detailed'
            },
            PersonaCluster.PROFESSIONAL.value: {
                'keywords': ['professional', 'policy', 'compliance', 'regulation', 'standard', 'procedure', 'process', 'management'],
                'indicators': ['professional', 'manager', 'coordinator', 'specialist', 'officer'],
                'sections': ['procedures', 'policies', 'compliance', 'standards', 'requirements', 'guidelines'],
                'complexity': 'balanced',
                'detail_level': 'detailed'
            }
        }
    
    def _load_job_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load job patterns for intelligent job analysis"""
        return {
            'planning': {
                'keywords': ['plan', 'organize', 'schedule', 'arrange', 'coordinate', 'prepare'],
                'categories': ['planning', 'organization', 'coordination'],
                'priority_sections': ['guide', 'steps', 'recommendations', 'tips', 'how_to']
            },
            'analysis': {
                'keywords': ['analyze', 'examine', 'evaluate', 'assess', 'review', 'study'],
                'categories': ['analysis', 'evaluation', 'research'],
                'priority_sections': ['data', 'results', 'findings', 'analysis', 'methodology']
            },
            'creation': {
                'keywords': ['create', 'build', 'develop', 'design', 'make', 'generate'],
                'categories': ['development', 'creation', 'design'],
                'priority_sections': ['implementation', 'design', 'specifications', 'guide', 'steps']
            },
            'management': {
                'keywords': ['manage', 'oversee', 'control', 'supervise', 'administer', 'handle'],
                'categories': ['management', 'administration', 'oversight'],
                'priority_sections': ['procedures', 'policies', 'guidelines', 'standards', 'requirements']
            },
            'learning': {
                'keywords': ['learn', 'understand', 'study', 'master', 'grasp', 'comprehend'],
                'categories': ['learning', 'education', 'understanding'],
                'priority_sections': ['introduction', 'concepts', 'examples', 'summary', 'basics']
            },
            'compliance': {
                'keywords': ['comply', 'conform', 'adhere', 'follow', 'meet', 'satisfy'],
                'categories': ['compliance', 'regulation', 'standards'],
                'priority_sections': ['requirements', 'standards', 'compliance', 'regulations', 'policies']
            }
        }
    
    def _load_section_patterns(self) -> Dict[str, float]:
        """Load section patterns for priority scoring"""
        return {
            'summary': 0.8, 'overview': 0.7, 'introduction': 0.7, 'conclusion': 0.8,
            'recommendations': 0.9, 'results': 0.8, 'findings': 0.8, 'analysis': 0.7,
            'methodology': 0.6, 'data': 0.7, 'research': 0.7, 'study': 0.7,
            'guide': 0.8, 'tips': 0.8, 'steps': 0.8, 'how_to': 0.8,
            'examples': 0.7, 'practice': 0.6, 'concepts': 0.7, 'basics': 0.7,
            'requirements': 0.8, 'standards': 0.8, 'compliance': 0.8, 'policies': 0.7,
            'procedures': 0.7, 'guidelines': 0.7, 'specifications': 0.7, 'technical': 0.6
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / "configs" / "model_config_1b.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('persona_intelligence', {})
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'confidence_threshold': 0.3,
            'max_keywords': 20,
            'enable_semantic_matching': True,
            'enable_adaptive_learning': False
        }
    
    def parse_persona(self, persona_data: Union[str, Dict[str, Any]]) -> PersonaProfile:
        """Dynamically parse any persona from input data"""
        try:
            if isinstance(persona_data, str):
                return self._parse_string_persona(persona_data)
            elif isinstance(persona_data, dict):
                return self._parse_dict_persona(persona_data)
            else:
                raise ValueError("Invalid persona_data type. Expected string or dictionary.")
                
        except Exception as e:
            self.logger.error(f"Error parsing persona: {e}")
            # Return a basic dynamic persona as fallback
            return self._create_default_persona(str(persona_data))
    
    def _parse_string_persona(self, persona_str: str) -> PersonaProfile:
        """Parse persona from string using intelligent pattern matching"""
        persona_lower = persona_str.lower().strip()
        
        # Detect persona cluster using domain patterns
        detected_cluster = self._detect_persona_cluster(persona_lower)
        cluster_info = self.domain_patterns.get(detected_cluster.value, {})
        
        # Extract keywords from the persona string
        persona_keywords = self._extract_smart_keywords(persona_lower)
        
        # Generate expertise areas based on the persona
        expertise_areas = self._generate_expertise_areas(persona_lower, detected_cluster)
        
        # Create dynamic persona profile
        persona_profile = PersonaProfile(
            persona_id=f"dynamic_{persona_lower.replace(' ', '_')}",
            original_persona=persona_str,
            persona_type=PersonaType.DYNAMIC,
            persona_cluster=detected_cluster,
            expertise_areas=expertise_areas,
            preferred_sections=cluster_info.get('sections', ['summary', 'overview', 'main']),
            detail_level=cluster_info.get('detail_level', 'detailed'),
            max_reading_time=self._estimate_reading_time(detected_cluster),
            job_categories=[],  # Will be filled by job analysis
            keywords=persona_keywords + cluster_info.get('keywords', []),
            exclusion_keywords=self._generate_exclusion_keywords(detected_cluster),
            complexity_preference=cluster_info.get('complexity', 'balanced')
        )
        
        self.logger.info(f"Dynamically parsed persona: '{persona_str}' -> {detected_cluster.value}")
        return persona_profile
    
    def _parse_dict_persona(self, persona_dict: Dict[str, Any]) -> PersonaProfile:
        """Parse persona from dictionary"""
        persona_role = persona_dict.get("role", "")
        return self._parse_string_persona(persona_role)
    
    def _detect_persona_cluster(self, persona_str: str) -> PersonaCluster:
        """Intelligently detect persona cluster from string"""
        best_cluster = PersonaCluster.DYNAMIC
        best_score = 0
        
        for cluster_name, cluster_info in self.domain_patterns.items():
            score = 0
            
            # Check keyword matches
            keywords = cluster_info.get('keywords', [])
            indicators = cluster_info.get('indicators', [])
            
            for keyword in keywords:
                if keyword in persona_str:
                    score += 2
            
            for indicator in indicators:
                if indicator in persona_str:
                    score += 3
            
            if score > best_score:
                best_score = score
                try:
                    best_cluster = PersonaCluster(cluster_name)
                except ValueError:
                    continue
        
        return best_cluster if best_score > 0 else PersonaCluster.DYNAMIC
    
    def _extract_smart_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        words = text.lower().split()
        keywords = []
        
        # Filter meaningful words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}"\'-')
            if len(clean_word) > 2 and clean_word not in stopwords and clean_word.isalpha():
                keywords.append(clean_word)
        
        return keywords[:10]  # Limit to top 10
    
    def _generate_expertise_areas(self, persona_str: str, cluster: PersonaCluster) -> List[str]:
        """Generate relevant expertise areas based on persona and cluster"""
        base_areas = []
        
        if cluster == PersonaCluster.ANALYTICAL:
            base_areas = ["research", "analysis", "data interpretation", "methodology"]
        elif cluster == PersonaCluster.BUSINESS:
            base_areas = ["business strategy", "market analysis", "financial planning", "decision making"]
        elif cluster == PersonaCluster.TECHNICAL:
            base_areas = ["technical implementation", "system design", "problem solving", "optimization"]
        elif cluster == PersonaCluster.EDUCATIONAL:
            base_areas = ["learning", "knowledge acquisition", "skill development", "comprehension"]
        elif cluster == PersonaCluster.SERVICE:
            base_areas = ["planning", "organization", "coordination", "service delivery"]
        elif cluster == PersonaCluster.PROFESSIONAL:
            base_areas = ["professional standards", "compliance", "process management", "quality assurance"]
        else:
            base_areas = ["general expertise", "professional knowledge", "task completion", "problem solving"]
        
        # Add persona-specific keywords as expertise
        persona_words = self._extract_smart_keywords(persona_str)
        for word in persona_words[:3]:  # Add top 3 persona words
            if word not in ' '.join(base_areas):
                base_areas.append(word)
        
        return base_areas
    
    def _estimate_reading_time(self, cluster: PersonaCluster) -> int:
        """Estimate reading time based on cluster"""
        time_mapping = {
            PersonaCluster.ANALYTICAL: 300,
            PersonaCluster.BUSINESS: 180,
            PersonaCluster.TECHNICAL: 240,
            PersonaCluster.EDUCATIONAL: 150,
            PersonaCluster.SERVICE: 180,
            PersonaCluster.PROFESSIONAL: 200,
            PersonaCluster.DYNAMIC: 180
        }
        return time_mapping.get(cluster, 180)
    
    def _generate_exclusion_keywords(self, cluster: PersonaCluster) -> List[str]:
        """Generate exclusion keywords based on cluster"""
        exclusions = {
            PersonaCluster.ANALYTICAL: ["advertisement", "promotion", "sales", "marketing"],
            PersonaCluster.BUSINESS: ["technical_details", "implementation", "code", "programming"],
            PersonaCluster.TECHNICAL: ["marketing", "sales", "financial", "business_strategy"],
            PersonaCluster.EDUCATIONAL: ["advanced", "complex", "technical_jargon", "specialized"],
            PersonaCluster.SERVICE: ["technical", "complex_analysis", "academic", "research"],
            PersonaCluster.PROFESSIONAL: ["informal", "casual", "entertainment", "personal"]
        }
        return exclusions.get(cluster, [])
    
    def _create_default_persona(self, persona_str: str) -> PersonaProfile:
        """Create a default persona when parsing fails"""
        return PersonaProfile(
            persona_id=f"default_{hash(persona_str) % 1000}",
            original_persona=persona_str,
            persona_type=PersonaType.DYNAMIC,
            persona_cluster=PersonaCluster.DYNAMIC,
            expertise_areas=["general", "professional"],
            preferred_sections=["summary", "overview", "main", "important"],
            detail_level="detailed",
            max_reading_time=180,
            keywords=self._extract_smart_keywords(persona_str.lower()),
            exclusion_keywords=[]
        )
    
    def analyze_job_to_be_done(self, job_description: str, original_persona: str = "") -> Dict[str, Any]:
        """Dynamically analyze any job-to-be-done to extract requirements"""
        job_lower = job_description.lower()
        
        # Detect job patterns
        detected_patterns = self._detect_job_patterns(job_lower)
        
        # Extract priority keywords from job description
        job_keywords = self._extract_smart_keywords(job_lower)
        
        # Combine with pattern-based keywords
        priority_keywords = job_keywords.copy()
        for pattern_info in detected_patterns.values():
            priority_keywords.extend(pattern_info.get('keywords', []))
        
        # Remove duplicates and limit
        priority_keywords = list(set(priority_keywords))[:15]
        
        # Extract job categories
        job_categories = []
        for pattern_name, pattern_info in detected_patterns.items():
            job_categories.extend(pattern_info.get('categories', []))
        
        # If no patterns detected, infer from keywords
        if not job_categories:
            job_categories = self._infer_job_categories(job_lower)
        
        # Extract detail level requirement
        detail_level = self._extract_detail_level(job_lower)
        
        # Extract time sensitivity
        time_sensitive = self._extract_time_sensitivity(job_lower)
        
        # Generate priority sections based on detected patterns
        priority_sections = []
        for pattern_info in detected_patterns.values():
            priority_sections.extend(pattern_info.get('priority_sections', []))
        
        if not priority_sections:
            priority_sections = ['summary', 'overview', 'main', 'important']
        
        return {
            "job_categories": list(set(job_categories)),
            "priority_keywords": priority_keywords,
            "priority_sections": list(set(priority_sections)),
            "time_sensitive": time_sensitive,
            "detail_level": detail_level,
            "original_description": job_description,
            "original_persona": original_persona,  # Store original persona
            "detected_patterns": list(detected_patterns.keys())
        }
    
    def _detect_job_patterns(self, job_str: str) -> Dict[str, Dict[str, Any]]:
        """Detect job patterns from description"""
        detected = {}
        
        for pattern_name, pattern_info in self.job_patterns.items():
            keywords = pattern_info.get('keywords', [])
            
            # Check if any keywords match
            matches = sum(1 for keyword in keywords if keyword in job_str)
            
            if matches > 0:
                detected[pattern_name] = pattern_info
        
        return detected
    
    def _infer_job_categories(self, job_str: str) -> List[str]:
        """Infer job categories when no patterns match"""
        categories = []
        
        # Action-based inference
        if any(word in job_str for word in ['create', 'build', 'develop', 'make', 'generate']):
            categories.append('creation')
        if any(word in job_str for word in ['analyze', 'examine', 'evaluate', 'assess']):
            categories.append('analysis')
        if any(word in job_str for word in ['plan', 'organize', 'schedule', 'prepare']):
            categories.append('planning')
        if any(word in job_str for word in ['manage', 'oversee', 'coordinate', 'handle']):
            categories.append('management')
        if any(word in job_str for word in ['learn', 'understand', 'study', 'master']):
            categories.append('learning')
        if any(word in job_str for word in ['comply', 'follow', 'adhere', 'meet']):
            categories.append('compliance')
        
        return categories if categories else ['general']
    
    def _extract_detail_level(self, job_str: str) -> str:
        """Extract detail level requirement from job description"""
        if any(word in job_str for word in ['comprehensive', 'detailed', 'thorough', 'in-depth', 'complete']):
            return 'comprehensive'
        elif any(word in job_str for word in ['overview', 'summary', 'brief', 'quick', 'basic']):
            return 'summary'
        else:
            return 'detailed'
    
    def _extract_time_sensitivity(self, job_str: str) -> bool:
        """Extract time sensitivity from job description"""
        return any(word in job_str for word in ['urgent', 'quickly', 'asap', 'immediate', 'fast', 'rush'])
    
    def get_section_priorities(self, persona: PersonaProfile, job_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Dynamically generate section priorities based on persona and job"""
        priorities = {}
        
        # Start with base section priorities
        for section, score in self.section_patterns.items():
            priorities[section] = score
        
        # Boost priorities based on persona preferences
        for section in persona.preferred_sections:
            if section in priorities:
                priorities[section] = min(1.0, priorities[section] + 0.2)
            else:
                priorities[section] = 0.8
        
        # Boost priorities based on job priority sections
        priority_sections = job_analysis.get('priority_sections', [])
        for section in priority_sections:
            if section in priorities:
                priorities[section] = min(1.0, priorities[section] + 0.3)
            else:
                priorities[section] = 0.9
        
        # Adjust based on job categories
        job_categories = job_analysis.get('job_categories', [])
        for category in job_categories:
            if category == 'analysis':
                priorities.update({'analysis': 0.9, 'results': 0.9, 'findings': 0.8, 'data': 0.8})
            elif category == 'planning':
                priorities.update({'guide': 0.9, 'steps': 0.9, 'recommendations': 0.8, 'tips': 0.8})
            elif category == 'compliance':
                priorities.update({'requirements': 0.9, 'standards': 0.9, 'compliance': 1.0, 'policies': 0.8})
            elif category == 'learning':
                priorities.update({'concepts': 0.9, 'examples': 0.8, 'basics': 0.8, 'introduction': 0.8})
        
        return priorities
    
    def to_dict(self, persona: PersonaProfile) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return asdict(persona)