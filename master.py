import os
import re
import PyPDF2
import docx
import nltk
import numpy as np
import pandas as pd
from termcolor import colored
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime
import pytz
import hashlib
import pickle
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
import argparse

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuration
CACHE_DIR = ".search_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@dataclass
class SearchResult:
    file_path: str
    content: str
    score: float
    page_num: Optional[int] = None
    position: Optional[Tuple[int, int]] = None
    metadata: Optional[Dict] = None

class AdvancedDocumentSearch:
    def __init__(self, model_name='all-mpnet-base-v2', cache_embeddings=True):
        self.model = SentenceTransformer(model_name)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.cache_embeddings = cache_embeddings
        self.document_cache = {}
        self.query_history = []
        
        # Initialize with default settings
        self.settings = {
            'highlight_color': 'yellow',
            'result_limit': 10,
            'similarity_threshold': 0.3,
            'chunk_mode': 'paragraph',
            'show_progress': True,
            'enable_cache': True
        }

    def _get_cache_key(self, file_path: str) -> str:
        """Generate unique cache key for file"""
        file_stat = os.stat(file_path)
        return hashlib.md5(f"{file_path}-{file_stat.st_mtime}-{file_stat.st_size}".encode()).hexdigest()

    def _load_from_cache(self, file_path: str) -> Optional[Dict]:
        """Load cached embeddings and metadata"""
        if not self.settings['enable_cache']:
            return None
            
        cache_key = self._get_cache_key(file_path)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, file_path: str, data: Dict):
        """Save embeddings and metadata to cache"""
        if not self.settings['enable_cache']:
            return
            
        cache_key = self._get_cache_key(file_path)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def _preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,;!?]', ' ', text)
        
        # Lemmatization and stopword removal
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

    def extract_text(self, file_path: str) -> Dict:
        """Extract text from various file formats with metadata"""
        file_ext = os.path.splitext(file_path)[1].lower()
        result = {
            'text': '',
            'metadata': {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)),
                'chunks': []
            }
        }
        # Ensure the variables are accessed
        print(f"Processing file with extension: {file_ext}")
        return result

        try:
            if file_ext == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    result['metadata']['page_count'] = len(reader.pages)
                    
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ''
                        result['text'] += page_text + '\n'
                        result['metadata']['chunks'].append({
                            'start': len(result['text']),
                            'end': len(result['text']) + len(page_text),
                            'page': i + 1
                        })
                        
            elif file_ext == '.docx':
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                result['text'] = '\n\n'.join(paragraphs)
                result['metadata']['paragraph_count'] = len(paragraphs)
                
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    result['text'] = f.read()
                    
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            result['metadata']['word_count'] = len(word_tokenize(result['text']))
            result['metadata']['char_count'] = len(result['text'])
            
        except Exception as e:
            print(colored(f"Error processing {file_path}: {str(e)}", 'red'))
            result['error'] = str(e)
            
        return result

    def chunk_text(self, text: str, metadata: Dict, mode: str = 'paragraph') -> List[Dict]:
        """Split text into chunks with position tracking"""
        chunks = []
        
        if mode == 'sentence':
            sentences = sent_tokenize(text)
            pos = 0
            for sent in sentences:
                chunks.append({
                    'text': sent,
                    'start': pos,
                    'end': pos + len(sent),
                    'type': 'sentence'
                })
                pos += len(sent) + 1
                
        elif mode == 'paragraph':
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            pos = 0
            for para in paragraphs:
                chunks.append({
                    'text': para,
                    'start': pos,
                    'end': pos + len(para),
                    'type': 'paragraph'
                })
                pos += len(para) + 2
                
        elif mode == 'page' and 'page_count' in metadata:
            # Use PDF page boundaries
            for chunk_info in metadata['chunks']:
                chunk_text = text[chunk_info['start']:chunk_info['end']]
                chunks.append({
                    'text': chunk_text,
                    'start': chunk_info['start'],
                    'end': chunk_info['end'],
                    'page': chunk_info['page'],
                    'type': 'page'
                })
                
        else:
            # Fallback to fixed-size chunks
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunks.append({
                    'text': text[i:i+chunk_size],
                    'start': i,
                    'end': min(i+chunk_size, len(text)),
                    'type': 'fixed'
                })
                
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.update({
                'file_path': metadata['file_path'],
                'file_size': metadata['file_size'],
                'modified_time': metadata['modified_time']
            })
            
        return chunks

    def process_document(self, file_path: str) -> List[Dict]:
        """Process a document and return chunked data with embeddings"""
        # Check cache first
        cached_data = self._load_from_cache(file_path)
        if cached_data:
            return cached_data['chunks']
            
        # Extract and process document
        doc_data = self.extract_text(file_path)
        if 'error' in doc_data:
            return []
            
        chunks = self.chunk_text(doc_data['text'], doc_data['metadata'], self.settings['chunk_mode'])
        
        # Generate embeddings for each chunk
        texts = [self._preprocess_text(chunk['text']) for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=self.settings['show_progress'])
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            
        # Cache the results
        if self.cache_embeddings:
            self._save_to_cache(file_path, {
                'metadata': doc_data['metadata'],
                'chunks': chunks
            })
            
        return chunks

    def search_single_file(self, file_path: str, query: str) -> List[SearchResult]:
        """Search within a single file"""
        chunks = self.process_document(file_path)
        if not chunks:
            return []
            
        # Process query
        query_embedding = self.model.encode(self._preprocess_text(query))
        
        # Calculate similarities
        chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Collect results above threshold
        results = []
        for i, score in enumerate(similarities):
            if score >= self.settings['similarity_threshold']:
                chunk = chunks[i]
                results.append(SearchResult(
                    file_path=chunk['file_path'],
                    content=chunk['text'],
                    score=float(score),
                    page_num=chunk.get('page'),
                    position=(chunk['start'], chunk['end']),
                    metadata=chunk
                ))
                
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.settings['result_limit']]

    def search_directory(self, directory: str, query: str) -> Dict[str, List[SearchResult]]:
        """Search all supported documents in a directory"""
        results = defaultdict(list)
        supported_ext = ['.pdf', '.docx', '.txt']
        
        # Record query history
        self.query_history.append({
            'query': query,
            'time': datetime.now(pytz.utc),
            'directory': directory
        })
        
        # Process each file
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_ext):
                    file_path = os.path.join(root, file)
                    try:
                        file_results = self.search_single_file(file_path, query)
                        if file_results:
                            results[file_path] = file_results
                    except Exception as e:
                        print(colored(f"Error searching {file_path}: {str(e)}", 'red'))
                        
        return dict(results)

    def highlight_match(self, text: str, query: str) -> str:
        """Highlight query terms in text"""
        query_terms = set(self._preprocess_text(query).split())
        words = word_tokenize(text)
        
        highlighted = []
        for word in words:
            lemma = self.lemmatizer.lemmatize(word.lower())
            if lemma in query_terms:
                highlighted.append(colored(word, self.settings['highlight_color'], attrs=['bold']))
            else:
                highlighted.append(word)
                
        return ' '.join(highlighted)

    def display_results(self, results: Dict[str, List[SearchResult]], query: str):
        """Display search results in a user-friendly format"""
        if not results:
            print(colored("\nNo matching results found.", 'yellow'))
            return
            
        print(colored(f"\n📊 Found relevant content in {len(results)} files:", 'green', attrs=['bold']))
        
        for file_path, file_results in results.items():
            print(colored(f"\n📂 File: {file_path}", 'blue'))
            print(colored(f"🔍 Found {len(file_results)} relevant sections:", 'cyan'))
            
            for result in file_results:
                print(colored(f"\n⭐ Score: {result.score:.3f}", 'magenta'))
                if result.page_num:
                    print(colored(f"📄 Page: {result.page_num}", 'white'))
                
                highlighted = self.highlight_match(result.content, query)
                print(highlighted)
                
    def export_results(self, results: Dict[str, List[SearchResult]], format: str = 'csv'):
        """Export search results to file"""
        if not results:
            return False
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_data = []
        
        for file_path, file_results in results.items():
            for result in file_results:
                export_data.append({
                    'file_path': file_path,
                    'score': result.score,
                    'page': result.page_num,
                    'content': result.content,
                    'position_start': result.position[0] if result.position else None,
                    'position_end': result.position[1] if result.position else None,
                    'modified_time': result.metadata['modified_time'].isoformat() if result.metadata else None
                })
                
        df = pd.DataFrame(export_data)
        
        if format == 'csv':
            output_file = f"search_results_{timestamp}.csv"
            df.to_csv(output_file, index=False)
        elif format == 'excel':
            output_file = f"search_results_{timestamp}.xlsx"
            df.to_excel(output_file, index=False)
        elif format == 'json':
            output_file = f"search_results_{timestamp}.json"
            df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        print(colored(f"\n✅ Results exported to {output_file}", 'green'))
        return output_file

    def interactive_search(self):
        """Run an interactive search session"""
        print(colored("\n🔍 Advanced Document Search System", 'blue', attrs=['bold']))
        print(colored("=" * 50, 'blue'))
        
        while True:
            print("\nOptions:")
            print("1. Search directory")
            print("2. Change settings")
            print("3. View query history")
            print("4. Export results")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                directory = input("Enter directory path: ").strip()
                if not os.path.isdir(directory):
                    print(colored("Invalid directory path!", 'red'))
                    continue
                    
                query = input("Enter search query: ").strip()
                if not query:
                    print(colored("Query cannot be empty!", 'red'))
                    continue
                    
                results = self.search_directory(directory, query)
                self.display_results(results)
                self.last_results = results
                
            elif choice == '2':
                self.configure_settings()
                
            elif choice == '3':
                self.show_query_history()
                
            elif choice == '4':
                if not hasattr(self, 'last_results') or not self.last_results:
                    print(colored("No results to export!", 'yellow'))
                    continue
                    
                format = input("Export format (csv/excel/json): ").strip().lower()
                if format not in ['csv', 'excel', 'json']:
                    print(colored("Invalid format!", 'red'))
                    continue
                    
                self.export_results(self.last_results, format)
                
            elif choice == '5':
                print(colored("\nGoodbye!", 'green'))
                break
                
            else:
                print(colored("Invalid choice!", 'red'))

    def configure_settings(self):
        """Configure search settings interactively"""
        print(colored("\n⚙️ Current Settings:", 'blue'))
        for key, value in self.settings.items():
            print(f"{key}: {value}")
            
        print("\nWhich setting would you like to change?")
        print("(Enter setting name or 'done' to finish)")
        
        while True:
            setting = input("\nSetting name: ").strip().lower()
            if setting == 'done':
                break
                
            if setting not in self.settings:
                print(colored("Invalid setting name!", 'red'))
                continue
                
            current_value = self.settings[setting]
            new_value = input(f"Current value: {current_value}\nNew value: ").strip()
            
            # Convert to appropriate type
            if isinstance(current_value, bool):
                new_value = new_value.lower() in ('true', 'yes', '1')
            elif isinstance(current_value, int):
                try:
                    new_value = int(new_value)
                except ValueError:
                    print(colored("Must be an integer!", 'red'))
                    continue
            elif isinstance(current_value, float):
                try:
                    new_value = float(new_value)
                except ValueError:
                    print(colored("Must be a number!", 'red'))
                    continue
                    
            self.settings[setting] = new_value
            print(colored("Setting updated!", 'green'))

    def show_query_history(self):
        """Display search query history"""
        if not self.query_history:
            print(colored("\nNo query history found.", 'yellow'))
            return
            
        print(colored("\n📜 Search History:", 'blue'))
        for i, query in enumerate(self.query_history, 1):
            print(f"{i}. {query['time']} - {query['query']}")
            print(f"   Directory: {query['directory']}\n")

def main():
    parser = argparse.ArgumentParser(description='Advanced Document Search System')
    parser.add_argument('directory', nargs='?', help='Directory to search')
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--export', help='Export format (csv, excel, json)')
    args = parser.parse_args()

    search_engine = AdvancedDocumentSearch()
    
    if args.interactive:
        search_engine.interactive_search()
    elif args.directory and args.query:
        results = search_engine.search_directory(args.directory, args.query)
        search_engine.display_results(results)
        
        if args.export:
            search_engine.export_results(results, args.export)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()