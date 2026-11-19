import React, { useState } from "react";

const EXAMPLES = [
  "Where is the debug port on the CUAV V5 nano?",
  "The EKF reports height estimate divergence. What is the most likely cause?",
  "How do I start the gyroscope calibration in QGroundControl?",
  "Which parameters set the GPS antenna position relative to the centre of gravity?",
];

export default function App() {
  const [q, setQ] = useState(EXAMPLES[0]);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [zoom, setZoom] = useState(null);

  async function ask(question) {
    setBusy(true);
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    setRes(await r.json());
    setBusy(false);
  }

  return (
    <div className="page">
      <header>
        <div className="brand">PX4 User Guide · technician assistant</div>
        <div className="sub">Answers from the PX4 User Guide (CC BY 4.0), with the pictures that go with them</div>
      </header>
      <form onSubmit={(e) => { e.preventDefault(); ask(q); }} className="ask">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about wiring, setup, calibration, parameters..." />
        <button disabled={busy}>{busy ? "Searching…" : "Ask"}</button>
      </form>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e} onClick={() => { setQ(e); ask(e); }}>{e}</button>
        ))}
      </div>
      {res && (
        <main>
          <section className="answer">
            <div className="label">Answer <span className="mode">{res.mode}</span></div>
            <p>{res.answer}</p>
            <div className="source">Source: {res.answer_source}</div>
            <div className="lat">retrieval {res.latency_ms.text} ms · figures {res.latency_ms.images} ms</div>
          </section>
          <section className="figs">
            {res.images.length === 0 && <div className="none">No figure is relevant enough to be shown.</div>}
            {res.images.map((im) => (
              <figure key={im.id} onClick={() => setZoom(im)}>
                <img src={`data:${im.mime};base64,${im.png_base64}`} alt={im.caption} />
                <figcaption>
                  <b>{im.caption || im.id.split("/").pop()}</b>
                  <span className="score">{im.page} · confidence {Math.round(im.score * 100)} %</span>
                </figcaption>
              </figure>
            ))}
          </section>
          <section className="passages">
            <div className="label">Passages</div>
            {res.passages.map((p, i) => (
              <article key={p.id} className={i === 0 ? "top" : ""}>
                <div className="ph">{p.title} <span>{p.section} · {p.score.toFixed(2)}</span></div>
                <p>{p.text}</p>
              </article>
            ))}
          </section>
        </main>
      )}
      {zoom && (
        <div className="zoom" onClick={() => setZoom(null)}>
          <img src={`data:image/png;base64,${zoom.png_base64}`} alt={zoom.caption} />
        </div>
      )}
    </div>
  );
}
