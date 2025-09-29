import React, { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import './App.css';
import logo from './assets/logo2.png';
import sendIcon from './assets/send.png';
// import chatBubble2 from './assets/chat-bubble.png';
import chatBubble2 from './assets/new-message-5.png';


// === Backend API helpers ===
const API_BASE = (process.env.REACT_APP_API_BASE || "http://localhost:8000").replace(/\/$/, "");

async function apiAsk(question, reset_memory = false) {
  const r = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, reset_memory })
  });
  if (!r.ok) throw new Error("ask failed");
  return await r.json(); // { answer, citations }
}

async function apiFollowUp() {
  const r = await fetch(`${API_BASE}/followup`, { method: "POST" });
  if (!r.ok) throw new Error("followup failed");
  return await r.json();
}

async function apiReset() {
  const r = await fetch(`${API_BASE}/reset`, { method: "POST" });
  if (!r.ok) throw new Error("reset failed");
  return await r.json();
}
// === end API helpers ===


export default function DomainChatPoC() {
    const [conversations, setConversations] = useState(() => {
        try {
            const raw = localStorage.getItem("poc_domain_chat_history");
            return raw ? JSON.parse(raw) : [{ id: genId(), name: "Główna rozmowa", messages: [] }];
        } catch (e) {
            return [{ id: genId(), name: "Główna rozmowa", messages: [] }];
        }
    });
    const [activeConvIdx, setActiveConvIdx] = useState(0);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    
    const [canFollowUp, setCanFollowUp] = useState(false);
    const [followUpArmed, setFollowUpArmed] = useState(false);
    const [questionCount, setQuestionCount] = useState(0);
const endRef = useRef(null);
    const textareaRef = useRef(null);
    const MAX_LINES = 6;


    useEffect(() => {
        localStorage.setItem("poc_domain_chat_history", JSON.stringify(conversations));
    }, [conversations]);

    useEffect(() => scrollToBottom(), [conversations, activeConvIdx]);

    useEffect(() => {
        const ta = textareaRef.current;
        if (!ta) return;

        // ustawienia box-sizing i reset wysokości
        ta.style.boxSizing = "border-box";
        ta.style.height = "auto";

        const computed = window.getComputedStyle(ta);

        // pobierz lineHeight (jeśli 'normal' -> fallback do 1.2 * fontSize)
        let lineHeight = parseFloat(computed.lineHeight);
        if (Number.isNaN(lineHeight)) {
            const fontSize = parseFloat(computed.fontSize) || 16;
            lineHeight = fontSize * 1.2;
        }

        // paddingy i obramowania (żeby dokładnie policzyć wysokość)
        const paddingTop = parseFloat(computed.paddingTop) || 0;
        const paddingBottom = parseFloat(computed.paddingBottom) || 0;
        const borderTop = parseFloat(computed.borderTopWidth) || 0;
        const borderBottom = parseFloat(computed.borderBottomWidth) || 0;

        const maxHeightPx =
            Math.round(lineHeight * MAX_LINES + paddingTop + paddingBottom + borderTop + borderBottom);

        // desired = minimalna z naturalnej zawartości i maxHeight
        const desired = Math.min(ta.scrollHeight, maxHeightPx);

        ta.style.height = desired + "px";

        // jeśli zawartość przekracza max -> włącz scroll w textarea, inaczej ukryj
        ta.style.overflowY = ta.scrollHeight > maxHeightPx ? "auto" : "hidden";
    }, [input]);

    function genId() {
        return Math.random().toString(36).slice(2, 9);
    }

    const handleSendEnter = (e) => {
        if (e.nativeEvent && e.nativeEvent.isComposing) return;

        if (e.key === "Enter" && !e.shiftKey) {
            handleSend(e);
        }
    };
    function addMessage(role, text, extra = undefined) {
    setConversations(prev => {
        const copy = [...prev];
        copy[activeConvIdx] = { ...copy[activeConvIdx] };
        copy[activeConvIdx].messages = [
        ...copy[activeConvIdx].messages,
        { id: genId(), role, text, ts: Date.now(), ...extra },
        ];
        return copy;
    });
    }

    function handleSend(e) {
        e?.preventDefault();
        const question = input.trim();
        if (!question) return;
        addMessage("user", question);
        setInput("");
        simulateAssistantAnswer(question);
    }

    async function simulateAssistantAnswer(question) {
    try {
        setIsTyping(true);
        const res = await apiAsk(question, false);

        const answer = res?.answer ?? "Brak odpowiedzi.";
        const rows = Array.isArray(res?.data_rows) ? res.data_rows : [];
        const cols = Array.isArray(res?.data_columns) ? res.data_columns : [];

        // 1) tekst (animowane „pisanie”)
        typeAssistantText(answer);

        // 2) jeśli są wiersze tabeli — dołóż je jako osobny „blok”
        if (rows.length) {
        // po zakończeniu „pisania” też jest OK, ale najprościej od razu:
        addMessage("assistant", "[[TABLE]]", { table: { cols, rows } });
        }

        setQuestionCount(n => n + 1);
        setCanFollowUp(res?.module === "ustawa");
        setFollowUpArmed(false);
    } catch (e) {
        console.error(e);
        typeAssistantText("Błąd połączenia z backendem.");
    }
    }
    function typeAssistantText(fullText) {
        let i = 0;
        addMessage("assistant", "");
        const interval = setInterval(() => {
            i += Math.ceil(Math.random() * 3);
            const slice = fullText.slice(0, i);
            setConversations(prev => {
                const copy = [...prev];
                const msgs = [...copy[activeConvIdx].messages];
                const lastIdx = msgs.length - 1;
                if (msgs[lastIdx].role === "assistant") {
                    msgs[lastIdx] = { ...msgs[lastIdx], text: slice };
                }
                copy[activeConvIdx] = { ...copy[activeConvIdx], messages: msgs };
                return copy;
            });
            if (i >= fullText.length) {
                clearInterval(interval);
                setIsTyping(false);
            }
        }, 35);
    }

    async function armFollowUp() {
        try {
            await apiFollowUp();
            setFollowUpArmed(true);
        } catch (e) {
            console.error(e);
        }
    }

    function newConversation() {
        setConversations(prev => [...prev, { id: genId(), name: "Rozmowa " + (prev.length + 1), messages: [] }]);
        setActiveConvIdx(conversations.length);
        apiReset().catch(e => console.warn("apiReset failed", e));
    }

    function clearActiveConversation() {
        setConversations(prev => {
            const copy = [...prev];
            copy[activeConvIdx] = { ...copy[activeConvIdx], messages: [] };
            return copy;
        });
        apiReset().catch(e => console.warn("apiReset failed", e));
    }

    function scrollToBottom() {
        endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    const activeConv = conversations[activeConvIdx] || { messages: [] };
    const msgs = activeConv.messages || [];
    const lastAssistantIdx = (() => {
    for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") return i;
    }
    return -1;
    })();

    return (
        <div className="relative h-screen font-sans">
            {/* Tło z pola */}
            <div
                className="absolute inset-0 bg-cover bg-center absolute-background"
            />
            {/* Nakładka półprzezroczysta */}
            <div className="absolute inset-0" />

            {/* Właściwa zawartość czatu */}
            <div className="relative grid grid-cols-12 h-full">
                {/* Sidebar */}
                <aside className="col-span-3 border-slate-200 p-4 shadow-lg flex flex-col bg-white/80 overflow-y-auto">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-semibold text-slate-800">
                            <img width={28} height={28} src={logo} alt="logo" />
                            <span> KRUS AI</span>
                        </h2>
                        <button
                            className="new-message text-sm px-3 py-1 bg-brand hover:bg-brand-dark transition text-white rounded-lg shadow"
                            onClick={newConversation}
                        >
                            <img src={chatBubble2} width={22} height={22} />
                        </button>
                    </div>

                    <div className="space-y-2 overflow-auto flex-1 pr-1 chat-lists">
                        {conversations.map((c, idx) => (
                            <button
                                key={c.id}
                                onClick={() => setActiveConvIdx(idx)}
                                className={`w-full text-left p-3 rounded-lg flex items-center justify-between transition ${
                                    idx === activeConvIdx
                                        ? "bg-brand-light border-l-4 border-brand-dark"
                                        : "hover:bg-slate-50"
                                }`}
                            >
                                <div>
                                    <div className="font-medium text-sm text-slate-800">{c.name}</div>
                                    <div className="text-xs text-slate-500">{c.messages.length} wiadomości</div>
                                </div>
                                <div className="text-xs text-slate-400">
                                    {c.messages.length ? formatAgo(c.messages[c.messages.length - 1].ts) : "—"}
                                </div>
                            </button>
                        ))}
                    </div>

                    <div className="mt-6 text-xs text-slate-500">
                        Asystent udziela informacji w zakresie ustawy o ubezpieczeniu społecznym rolników. Odpowiedzi nie stanowią porady prawnej.
                    </div>
                </aside>

                {/* Main chat */}
                {activeConv.messages.length > 0 ? (
                    <main className="flex-1 overflow-y-auto col-span-9 flex flex-col">
                        <header className="px-6 py-4 border-slate-200   flex items-center justify-between bg-white/40">
                            <div>
                                <h3 className="text-lg font-semibold text-slate-800">{activeConv.name}</h3>
                            </div>
                        </header>

                        <section className="flex-1 p-6 overflow-y-auto  from-white to-slate-50 chat-messages bg-white/40">
                            <div className="max-w-3xl mx-auto space-y-4">
                                <AnimatePresence initial={false}>
                                    {activeConv.messages.map((msg, idx) => (
                                    <motion.div
                                        key={msg.id}
                                        initial={{ opacity: 0, y: 6 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -6 }}
                                    >
                                        <MessageBubble
                                        msg={msg}
                                        isLastAssistant={idx === lastAssistantIdx}
                                        showFollowUp={canFollowUp && questionCount >= 1 && !followUpArmed}
                                        onFollowUp={armFollowUp}
                                        />
                                    </motion.div>
                                    ))}
                                </AnimatePresence>

                                {isTyping && (
                                    <div className="flex items-start animate-pulse">
                                        <div className="w-9 h-9 rounded-full bg-green-100 flex items-center justify-center text-xs text-brand-dark font-bold">
                                            A
                                        </div>
                                        <div className="ml-3 px-4 py-2 bg-slate-100 rounded-xl text-sm shadow-sm italic text-gray-500">Pisze...</div>
                                    </div>
                                )}
                                <div ref={endRef} />
                            </div>
                        </section>

                        <form
                            onSubmit={handleSend}
                            className="px-6 py-4  border-slate-200  shadow-inner shrink-0 bg-white/40"
                        >
                            <div className="max-w-3xl mx-auto flex gap-3 ">
                            <textarea
                                ref={textareaRef}
                                value={input}
                                rows={1}
                                onKeyDown={handleSendEnter}
                                onChange={(e) => setInput(e.target.value)}
                                className="input-text border-green resize-none  flex-1 p-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-1 focus:ring-brand shadow-sm"
                                placeholder="Zadaj pytanie..."
                            />
                                <button
                                    type="submit"
                                    className="px-5 py-2 rounded-xl bg-brand hover:bg-brand-dark transition text-white font-medium shadow-md disabled:opacity-50"
                                    disabled={!input.trim()}
                                >
                                    <img height={20} width={20} src={sendIcon} alt="send" />
                                </button>
                            </div>
                        </form>
                    </main>
                ) : (
                    <main className="flex-1 overflow-y-auto col-span-9 flex flex-col">
                        <motion.section
                            key="welcome"
                            className="flex-1 flex flex-col items-center welcome-stext justify-center text-center px-6 bg-white/40"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                        >
                            <h1 className="main-text text-2xl font-bold  mb-4">
                                👋 Witaj! Tu Asystent Kasy Rolniczego Ubezpieczenia Społecznego
                            </h1>
                            <p className="main-text text-semibold mb-8 max-w-screen-lg">
                                Udzielam informacji w zakresie ustawy o ubezpieczeniu społecznym rolników, w tym kogo obejmuje ubezpieczenie i jakie świadczenia przysługują. Mogę także odpowiadać na pytania dotyczące danych statystycznych, takich jak kwoty i liczba świadczeń w latach 2024–2025.<br/>
                                Udzielane informacje mają charakter ogólny i nie stanowią porady prawnej ani indywidualnej interpretacji przepisów.                            </p>
                            <div className="w-full max-w-2xl">
                                <form onSubmit={handleSend} className="flex gap-3">
                                    <textarea
                                        ref={textareaRef}
                                        value={input}
                                        rows={1}
                                        onKeyDown={handleSendEnter}
                                        onChange={(e) => setInput(e.target.value)}
                                        className="input-text resize-none flex-1 p-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-1 focus:ring-brand shadow-sm"
                                        placeholder="Zadaj pytanie..."
                                    />
                                    <button
                                        type="submit"
                                        className="px-5 py-2 rounded-xl bg-brand hover:bg-brand-dark transition text-white font-medium shadow-md disabled:opacity-50"
                                        disabled={!input.trim()}
                                    >
                                        <img height={20} width={20} src={sendIcon} alt="send" />
                                    </button>
                                </form>
                            </div>
                        </motion.section>
                    </main>
                )}

            </div>
        </div>
    );
}

function MessageBubble({ msg, isLastAssistant = false, showFollowUp = false, onFollowUp }) {
  if (msg.role === "user") {
    return (
      <div className="flex items-start justify-end">
        <div className="max-w-[70%]">
          <div className="text-xs text-slate-400 text-right mb-1">Ty</div>
          <div className="bg-brand text-white px-4 py-2 rounded-xl text-sm shadow-md break-words whitespace-pre-wrap">
            {msg.text}
          </div>
        </div>
      </div>
    );
  }

  const safeText = (msg.text ?? "").replace(/\\n/g, "\n");

  return (
    <div className="flex items-start">
      <div className="w-9 h-9 rounded-full bg-brand-light flex items-center justify-center text-xs ai-text font-bold">A</div>
      <div className="ml-3 max-w-[70%]">
        <div className="text-xs text-slate-400 mb-1">Asystent</div>

        {/* Tekst odpowiedzi */}
        {safeText && safeText !== "[[TABLE]]" && (
          <div className="bg-white px-4 py-2 rounded-xl border border-slate-200 text-sm shadow-sm break-words whitespace-pre-wrap">
            {safeText}
          </div>
        )}

        {/* Tabela (jeśli jest) */}
        {msg.table && Array.isArray(msg.table.rows) && msg.table.rows.length > 0 && (
          <div className="mt-2 bg-white rounded-xl border border-slate-200 shadow-sm overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  {(msg.table.cols ?? Object.keys(msg.table.rows[0] ?? {})).map((c) => (
                    <th key={c} className="px-3 py-2 text-left font-semibold text-slate-700 border-b">
                      {headerLabel(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {msg.table.rows.map((r, idx) => (
                  <tr key={idx} className="even:bg-slate-50/40">
                    {(msg.table.cols ?? Object.keys(r)).map((c) => (
                      <td key={c} className="px-3 py-2 text-slate-800 border-b align-top">
                        {formatCell(r[c], c)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Guzik follow-up: WYŚRODKOWANY, zielony jak "Wyślij", biała czcionka */}
        {isLastAssistant && showFollowUp && (
          <div className="mt-3">
            <button
              onClick={onFollowUp}
              className="mx-auto block px-5 py-2 rounded-xl bg-brand hover:bg-brand-dark transition text-white font-medium shadow-md focus:outline-none focus:ring-2 focus:ring-brand/40"
              title="Następne pytanie potraktuję jako dopytanie."
            >
              Chciałbym dopytać
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ładniejsze nagłówki
function headerLabel(key) {
  const map = {
    value: "wartość",
    dataset: "tabela",
    measure: "miara",
    type: "typ",
    okres: "okres",
    region: "region",
    ce: "ce",
    rrf: "rrf",
  };
  return map[key] ?? key;
}

// delikatne formatowanie wartości
function formatCell(v, key) {
  if (v == null) return "—";
  if ((key === "value" || key === "ce" || key === "rrf") && typeof v === "number") {
    // value: do 2 miejsc; ce/rrf: do 3
    const digits = key === "value" ? 2 : 3;
    return v.toLocaleString("pl-PL", { maximumFractionDigits: digits });
  }
  return String(v);
}


function formatAgo(ts) {
    if (!ts) return "—";
    const diff = Math.floor((Date.now() - ts) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return `${Math.floor(diff / 86400)}d`;
}
