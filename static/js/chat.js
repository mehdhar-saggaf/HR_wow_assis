const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("msg");

const SESSION_ID = (() => {
  let sid = localStorage.getItem("hr_session_id");
  if (!sid) {
    sid = Math.random().toString(36).slice(2);
    localStorage.setItem("hr_session_id", sid);
  }
  return sid;
})();

function bubble(text, who = "bot", citations = []) {
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  el.innerText = text;
  if (citations.length) {
    const refs = document.createElement("div");
    refs.className = "citations";
    refs.innerText = "المراجع: " + citations.map(c => `${c.doc_title}#${c.chunk}`).join(" ، ");
    el.appendChild(refs);
  }
  chat.appendChild(el);
  el.scrollIntoView({ behavior: "smooth" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  bubble(q, "user");
  input.value = "";
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, session_id: SESSION_ID }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    bubble(data.answer, "bot", data.citations || []);
  } catch (err) {
    bubble("حدث خطأ غير متوقع: " + err.message, "bot");
  }
});
