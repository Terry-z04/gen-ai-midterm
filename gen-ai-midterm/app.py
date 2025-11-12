#!/usr/bin/env python3
"""
Flask Web Application for UChicago MS-ADS RAG System
Production-ready for AWS EC2 deployment

Features:
- Advanced RAG with HyDE + RAG Fusion
- Streaming responses
- Production WSGI ready
- AWS EC2 compatible
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import json
import time
from datetime import datetime

# Import RAG systems
try:
    from advanced_rag import AdvancedRAG
    from qa_generator import QAGenerator
    from fallback_guardrail import FallbackGuardrail
    from config import Config
    from monitoring import MonitoringSystem
    SYSTEM_READY = True
except Exception as e:
    print(f"⚠ Error importing: {e}")
    SYSTEM_READY = False


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

# Initialize RAG system (lazy loading)
rag_system = None
qa_generator = None
guardrail_system = None
monitoring_system = None

def get_monitoring():
    """Lazy load monitoring system"""
    global monitoring_system
    if monitoring_system is None and SYSTEM_READY:
        try:
            monitoring_system = MonitoringSystem()
            print("✓ Monitoring system initialized")
        except Exception as e:
            print(f"✗ Failed to initialize monitoring: {e}")
    return monitoring_system

def get_rag_system():

    """Lazy load RAG system"""
    global rag_system
    if rag_system is None and SYSTEM_READY:
        try:
            rag_system = AdvancedRAG()
            print("✓ Advanced RAG system initialized")
        except Exception as e:
            print(f"✗ Failed to initialize RAG: {e}")
    return rag_system

def get_qa_generator():
    """Lazy load QA generator"""
    global qa_generator
    if qa_generator is None and SYSTEM_READY:
        try:
            qa_generator = QAGenerator(model="gpt-4o-mini")
            print("✓ QA Generator initialized")
        except Exception as e:
            print(f"✗ Failed to initialize QA: {e}")
    return qa_generator

def get_guardrail():
    """Lazy load guardrail system"""
    global guardrail_system
    if guardrail_system is None and SYSTEM_READY:
        try:
            guardrail_system = FallbackGuardrail(enable_logging=False)
            print("✓ Fallback Guardrail initialized")
        except Exception as e:
            print(f"✗ Failed to initialize guardrail: {e}")
    return guardrail_system


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint for AWS"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'system_ready': SYSTEM_READY
    })


@app.route('/api/query', methods=['POST'])
def query():
    """Main query endpoint with automatic fallback guardrail"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        method = data.get('method', 'advanced')  # 'advanced' or 'basic'
        model = data.get('model', 'gpt-4o-mini')
        use_fallback = data.get('use_fallback', True)  # Enable fallback by default
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get initial response using selected method
        if method == 'advanced':
            # Use advanced RAG
            rag = get_rag_system()
            if not rag:
                return jsonify({'error': 'RAG system not initialized'}), 500
            
            response = rag.answer_with_advanced_rag(
                question=question,
                top_k=8,
                model=model
            )
            initial_results = response.get('retrieval', {}).get('results', [])
        else:
            # Use basic QA
            qa = get_qa_generator()
            if not qa:
                return jsonify({'error': 'QA system not initialized'}), 500
            
            qa.model = model
            response = qa.answer_with_retrieval(question=question)
            initial_results = response.get('retrieval', {}).get('results', [])
        
        # Apply fallback guardrail if enabled
        if use_fallback:
            guardrail = get_guardrail()
            if guardrail:
                # Check if initial answer shows uncertainty
                is_uncertain, pattern = guardrail.detect_uncertainty(response.get('answer', ''))
                
                if is_uncertain:
                    # Trigger fallback with live data
                    fallback_response = guardrail.answer_with_fallback(
                        question=question,
                        initial_results=initial_results
                    )
                    
                    # Return guardrail response
                    return jsonify({
                        'success': True,
                        'question': question,
                        'answer': fallback_response.get('answer', ''),
                        'method': method,
                        'model': fallback_response.get('model', model),
                        'tokens': fallback_response.get('tokens', {}),
                        'retrieval_info': {
                            **response.get('retrieval', {}),
                            'fallback_triggered': True,
                            'fallback_success': fallback_response.get('fallback_success', False),
                            'live_sources': fallback_response.get('live_sources', []),
                            'uncertainty_pattern': pattern
                        },
                        'confidence': fallback_response.get('confidence', 'unknown'),
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        # Return original response
        return jsonify({
            'success': True,
            'question': question,
            'answer': response.get('answer', ''),
            'method': method,
            'model': response.get('model', model),
            'tokens': response.get('tokens', {}),
            'retrieval_info': response.get('retrieval', {}),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/query/stream', methods=['POST'])
def query_stream():
    """Streaming query endpoint"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        def generate():
            """Generator for streaming response"""
            try:
                # Get RAG system
                rag = get_rag_system()
                if not rag:
                    yield json.dumps({'error': 'System not ready'}) + '\n'
                    return
                
                # Get context (non-streaming)
                retrieval_response = rag.multi_method_retrieval(question, top_k=8)
                
                # Send retrieval info first
                yield json.dumps({
                    'type': 'retrieval',
                    'num_docs': len(retrieval_response['results']),
                    'method': retrieval_response['method']
                }) + '\n'
                
                # Format context
                context_parts = ["## Retrieved Context\n"]
                for i, result in enumerate(retrieval_response['results'], 1):
                    text = result.get('text', '').strip()
                    context_parts.append(f"### Source {i}\n{text}\n")
                context = '\n'.join(context_parts)
                
                # Stream answer
                qa = get_qa_generator()
                qa.use_streaming = True
                qa.model = "gpt-4o-mini"
                
                # Generate with streaming
                from openai import OpenAI
                client = OpenAI(api_key=Config.OPENAI_API_KEY)
                
                user_prompt = f"""Context:
{context}

Question: {question}

Please provide a helpful, accurate answer based ONLY on the context above."""
                
                messages = [
                    {"role": "system", "content": qa_generator.SYSTEM_PROMPT if qa_generator else "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]
                
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield json.dumps({
                            'type': 'token',
                            'content': chunk.choices[0].delta.content
                        }) + '\n'
                
                # Send completion
                yield json.dumps({'type': 'done'}) + '\n'
                
            except Exception as e:
                yield json.dumps({'type': 'error', 'error': str(e)}) + '\n'
        
        return Response(
            stream_with_context(generate()),
            mimetype='application/x-ndjson'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/methods')
def get_methods():
    """Get available retrieval methods"""
    return jsonify({
        'methods': [
            {
                'id': 'advanced',
                'name': 'Advanced RAG',
                'description': 'HyDE + RAG Fusion + Multi-Method',
                'recommended': True
            },
            {
                'id': 'basic',
                'name': 'Basic RAG',
                'description': 'Hybrid retrieval (Semantic + BM25)',
                'recommended': False
            }
        ],
        'models': [
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'recommended': True},
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'recommended': False}
        ]
    })


@app.route('/api/query/guardrail', methods=['POST'])
def query_with_guardrail():
    """
    Query with automatic Firecrawl fallback when knowledge base doesn't have info
    Detects uncertainty and automatically fetches live data
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get guardrail system
        guardrail = get_guardrail()
        if not guardrail:
            return jsonify({'error': 'Guardrail system not initialized'}), 500
        
        # First, get initial retrieval from standard system
        qa = get_qa_generator()
        if not qa:
            return jsonify({'error': 'QA system not initialized'}), 500
        
        # Retrieve with confidence router
        retrieval_response = qa.answer_with_retrieval(question=question)
        
        # Get initial results
        initial_results = retrieval_response.get('retrieval', {}).get('results', [])
        
        # Use guardrail with fallback
        response = guardrail.answer_with_fallback(
            question=question,
            initial_results=initial_results
        )
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': response.get('answer', ''),
            'fallback_triggered': response.get('fallback_triggered', False),
            'fallback_success': response.get('fallback_success', False),
            'confidence': response.get('confidence', 'unknown'),
            'initial_answer': response.get('initial_answer'),
            'live_sources': response.get('live_sources', []),
            'uncertainty_pattern': response.get('uncertainty_pattern'),
            'tokens': response.get('tokens', {}),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dashboard')
def dashboard():
    """Monitoring dashboard endpoint"""
    try:
        days = request.args.get('days', 7, type=int)
        
        monitoring = get_monitoring()
        if not monitoring:
            return jsonify({'error': 'Monitoring system not available'}), 500
        
        dashboard_data = monitoring.get_dashboard_data(days=days)
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    print("\n" + "="*60)
    print("UChicago MS-ADS RAG System - Web App")
    print("="*60)
    print("\nStarting development server...")
    print("Access at: http://localhost:5000")
    print("\nFor production (AWS EC2), use:")
    print("  gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
