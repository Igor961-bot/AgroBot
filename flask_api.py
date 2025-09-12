#!/usr/bin/env python3
"""
Flask API backend dla chatbota prawniczego KRUS
Minimalistyczne API które wywołuje twoją funkcję ask()
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os
import traceback

# Import twojej funkcji ask - dostosuj ścieżkę
from krus_final import ask

app = Flask(__name__, static_folder='.')
CORS(app)  # Pozwala na wywołania AJAX z przeglądarki

# Globalna zmienna do debugowania
DEBUG_MODE = True

@app.route('/')
def serve_frontend():
    """Serwuje frontend HTML"""
    return send_from_directory('.', 'index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """
    Główne API endpoint dla chatbota
    Przyjmuje JSON: {"question": "pytanie", "reset_memory": bool}
    Zwraca JSON: {"answer": "odpowiedź", "time": float, "status": "success"}
    """
    
    start_time = time.time()
    
    try:
        # Pobierz dane z requestu
        data = request.get_json()
        if not data:
            return jsonify({"error": "Brak danych JSON"}), 400
        
        question = data.get('question', '').strip()
        reset_memory = data.get('reset_memory', False)
        
        if not question:
            return jsonify({"error": "Pytanie nie może być puste"}), 400
        
        if DEBUG_MODE:
            print(f"\n{'='*50}")
            print(f"API Call - Pytanie: '{question}'")
            print(f"Reset memory: {reset_memory}")
            print(f"Start time: {start_time}")
        
        # Wywołaj twoją funkcję ask
        result = ask(question, reset_memory=reset_memory)
        
        ask_time = time.time()
        ask_duration = ask_time - start_time
        
        if DEBUG_MODE:
            print(f"ask() zakończone po: {ask_duration:.3f}s")
            print(f"Typ wyniku: {type(result)}")
        
        # Przetwórz wynik
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            debug_info = result.get("debug", {})
        else:
            answer = str(result)
            debug_info = {}
        
        # Podstawowe formatowanie (opcjonalne)
        formatted_answer = format_legal_response(answer)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if DEBUG_MODE:
            print(f"Całkowity czas API: {total_duration:.3f}s")
            print(f"Długość odpowiedzi: {len(formatted_answer)} znaków")
        
        # Zwróć JSON response
        response = {
            "answer": formatted_answer,
            "time": round(total_duration, 3),
            "ask_time": round(ask_duration, 3),
            "status": "success",
            "debug": debug_info if DEBUG_MODE else {}
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_time = time.time() - start_time
        error_message = str(e)
        
        if DEBUG_MODE:
            print(f"BŁĄD API po {error_time:.3f}s: {error_message}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": error_message,
            "time": round(error_time, 3),
            "status": "error",
            "traceback": traceback.format_exc() if DEBUG_MODE else None
        }), 500

def format_legal_response(answer):
    """
    Podstawowe formatowanie odpowiedzi prawniczej
    (opcjonalne - możesz to pominąć jeśli twoja funkcja ask() już formatuje)
    """
    if not answer:
        return answer
    
    formatted = answer
    
    # Zamień podstawowe formatowanie na HTML/markdown
    if 'Cytowane ustępy:' in formatted:
        formatted = formatted.replace('Cytowane ustępy:', '📖 **Podstawa prawna:**')
        formatted = formatted.replace('- [', '  📋 **[')
        formatted = formatted.replace('] Rozdz.', ']** Rozdział ')
        formatted = formatted.replace(' Art.', ' • Artykuł ')
        formatted = formatted.replace(' Ust.', ' • Ustęp ')
    
    if 'Odpowiedź:' in formatted:
        formatted = formatted.replace('Odpowiedź:', '\n💬 **Interpretacja prawna:**')
    
    # Sprawdź czy końcówka nie została ucięta
    if "*Mogę popełniać błędy" not in formatted and "skonsultuj się z" not in formatted:
        formatted += "\n\n*Mogę popełniać błędy, skonsultuj się z placówką KRUS w celu potwierdzenia informacji.*"
    
    return formatted

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint do sprawdzania stanu API"""
    try:
        # Sprawdź czy funkcja ask jest dostępna
        if 'ask' not in globals():
            return jsonify({
                "status": "error",
                "message": "Funkcja ask() nie jest dostępna",
                "available_functions": list(globals().keys())
            }), 500
        
        return jsonify({
            "status": "healthy",
            "message": "API działa poprawnie",
            "debug_mode": DEBUG_MODE,
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/test', methods=['POST'])
def api_test():
    """Endpoint testowy - szybka odpowiedź bez wywołania ask()"""
    start_time = time.time()
    
    data = request.get_json()
    question = data.get('question', 'test') if data else 'test'
    
    # Symulacja odpowiedzi
    time.sleep(0.1)  # Symulacja małego opóźnienia
    
    end_time = time.time()
    duration = end_time - start_time
    
    return jsonify({
        "answer": f"✅ **Test API zakończony pomyślnie!**\n\nTwoje pytanie: '{question}'\nCzas odpowiedzi: {duration:.3f}s\n\n*To jest odpowiedź testowa - nie wywołuje funkcji ask().*",
        "time": round(duration, 3),
        "status": "test_success"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint nie został znaleziony"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Wewnętrzny błąd serwera"}), 500

if __name__ == '__main__':
    print("🚀 FLASK API dla Chatbota Prawniczego KRUS")
    print("="*50)
    print("📡 Endpoints:")
    print("  GET  /                 - Frontend HTML")
    print("  POST /api/ask          - Główne API chatbota")
    print("  POST /api/test         - Test API (bez ask())")
    print("  GET  /api/health       - Health check")
    print()
    print("🔧 Konfiguracja:")
    print(f"  Debug mode: {DEBUG_MODE}")
    print(f"  CORS: Enabled")
    print(f"  Port: 5000")
    print()
    
    # Sprawdź czy funkcja ask jest dostępna
    try:
        if 'ask' in globals():
            print("✅ Funkcja ask() jest dostępna")
        else:
            print("⚠️  UWAGA: Funkcja ask() nie jest zaimportowana!")
            print("   Dodaj na początku pliku:")
            print("   from your_legal_chatbot_file import ask")
    except Exception as e:
        print(f"❌ Błąd sprawdzania ask(): {e}")
    
    print()
    print("🌐 Uruchamiam serwer na http://localhost:5000")
    print("📱 Frontend dostępny na http://localhost:5000")
    print("🔧 Ctrl+C aby zatrzymać")
    
    # Uruchom Flask z debugowaniem
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=DEBUG_MODE,
        threaded=True  # Pozwala na jednoczesne połączenia
    )