/**
 * Project-14-specific views: the product gallery, a product's predictions,
 * text search over photos and live name classification. Each draws one API
 * response.
 */

import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '@dsfs/ui';

interface Top { article_type: string; probability: number }
interface GalleryItem { id: number; name: string; actual: string; image_prediction: string; fusion_prediction: string }
interface Gallery { total: number; page: number; per_page: number; classes: string[]; items: GalleryItem[] }
interface Product { id: number; name: string; masked_name: string; fields: Record<string, string | null>; actual: string; predictions: { model: string; top: Top[] }[] }
interface Search { query: string; gallery: number; results: { id: number; name: string; actual: string; similarity: number }[] }
interface NameResult { name: string; masked_name: string; models: { model: string; top: Top[] }[] }

const input: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '10px 12px', font: 'inherit', color: 'var(--text-primary)',
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
};
const tile: React.CSSProperties = {
  background: 'var(--surface-1)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden',
  textAlign: 'left', padding: 0, font: 'inherit', color: 'var(--text-primary)', cursor: 'pointer',
};
const mark = (ok: boolean): React.CSSProperties => ({ color: ok ? 'var(--status-good)' : 'var(--status-critical)', fontWeight: 600 });

function Bars({ top }: { top: Top[] }) {
  return (
    <div style={{ display: 'grid', gap: 4 }}>
      {top.map((t) => (
        <div key={t.article_type} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 52px', gap: 8, alignItems: 'center', fontSize: 'var(--text-sm)' }}>
          <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.article_type}</span>
          <div style={{ background: 'var(--surface-2)', borderRadius: 3, height: 10 }}>
            <div style={{ width: `${t.probability * 100}%`, height: 10, borderRadius: 3, background: 'var(--series-1)' }} />
          </div>
          <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{(t.probability * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export function ProductDetail({ id }: { id: number }) {
  const [p, setP] = useState<Product | null>(null);
  useEffect(() => { apiGet<Product>(`/api/products/${id}`).then(setP).catch(() => setP(null)); }, [id]);
  if (!p) return null;
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'minmax(180px, 240px) 1fr', alignItems: 'start' }}>
      <img src={`/api/images/${p.id}`} alt={p.name} style={{ width: '100%', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }} />
      <div style={{ display: 'grid', gap: 'var(--space-3)' }}>
        <div>
          <div style={{ fontWeight: 600 }}>{p.name}</div>
          <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>
            Label words removed: “{p.masked_name}” · {Object.entries(p.fields).map(([k, v]) => `${k} ${v ?? '—'}`).join(' · ')}
          </div>
          <div style={{ fontSize: 'var(--text-sm)', marginTop: 4 }}>Actual article type: <strong>{p.actual}</strong></div>
        </div>
        <div style={{ display: 'grid', gap: 'var(--space-3)', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
          {p.predictions.map((m) => (
            <div key={m.model}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 4 }}>
                {m.model} · <span style={mark(m.top[0].article_type === p.actual)}>{m.top[0].article_type === p.actual ? 'right' : 'wrong'}</span>
              </div>
              <Bars top={m.top} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function GalleryView({ classes }: { classes: string[] }) {
  const [type, setType] = useState('');
  const [page, setPage] = useState(0);
  const [g, setG] = useState<Gallery | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => {
    apiGet<Gallery>(`/api/gallery?page=${page}${type ? `&article_type=${encodeURIComponent(type)}` : ''}`)
      .then((r) => { setG(r); setSelected((s) => s ?? r.items[0]?.id ?? null); }).catch(() => setG(null));
  }, [type, page]);
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center', flexWrap: 'wrap' }}>
        <select value={type} onChange={(e) => { setType(e.target.value); setPage(0); setSelected(null); }} style={{ ...input, width: 260 }} aria-label="Article type">
          <option value="">All article types</option>
          {classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        {g && <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>{g.total.toLocaleString()} test products · page {page + 1} of {Math.max(1, Math.ceil(g.total / g.per_page))}</span>}
        <button type="button" style={{ ...input, width: 'auto', cursor: 'pointer' }} disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
        <button type="button" style={{ ...input, width: 'auto', cursor: 'pointer' }} disabled={!g || (page + 1) * g.per_page >= g.total} onClick={() => setPage(page + 1)}>Next</button>
      </div>
      {selected !== null && <ProductDetail id={selected} />}
      {g && (
        <div style={{ display: 'grid', gap: 'var(--space-2)', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
          {g.items.map((it) => (
            <button key={it.id} type="button" onClick={() => setSelected(it.id)}
                    style={{ ...tile, outline: selected === it.id ? '2px solid var(--series-1)' : 'none' }}>
              <img src={`/api/images/${it.id}`} alt={it.name} loading="lazy" style={{ width: '100%', aspectRatio: '3 / 4', objectFit: 'cover', display: 'block' }} />
              <div style={{ padding: 6, fontSize: 11, lineHeight: 1.4 }}>
                <div style={{ fontWeight: 600 }}>{it.actual}</div>
                <div>image: <span style={mark(it.image_prediction === it.actual)}>{it.image_prediction}</span></div>
                <div>fusion: <span style={mark(it.fusion_prediction === it.actual)}>{it.fusion_prediction}</span></div>
              </div>
            </button>
          ))}
        </div>
      )}
      <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
        “fusion” is the late fusion of image, name with label words removed, and catalogue fields. Click a product for every model's top three.
      </p>
    </div>
  );
}

export function SearchView({ initial }: { initial: string }) {
  const [q, setQ] = useState(initial);
  const [r, setR] = useState<Search | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (query: string) => {
    if (query.trim().length < 2) return;
    setBusy(true);
    try { setR(await apiGet<Search>(`/api/search?q=${encodeURIComponent(query)}`)); } finally { setBusy(false); }
  };
  useEffect(() => { void run(initial); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <form onSubmit={(e) => { e.preventDefault(); void run(q); }} style={{ display: 'flex', gap: 'var(--space-2)' }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} style={input} aria-label="Describe a product" />
        <button type="submit" style={{ ...input, width: 'auto', cursor: 'pointer', fontWeight: 600 }}>{busy ? 'Searching…' : 'Search'}</button>
      </form>
      {r && (
        <>
          <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
            The {r.results.length} photos, of {r.gallery.toLocaleString()} test products, whose CLIP embedding is closest to “{r.query}”. Names are shown for reference; only the photos were searched.
          </p>
          <div style={{ display: 'grid', gap: 'var(--space-2)', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
            {r.results.map((it) => (
              <div key={it.id} style={{ ...tile, cursor: 'default' }}>
                <img src={`/api/images/${it.id}`} alt={it.name} loading="lazy" style={{ width: '100%', aspectRatio: '3 / 4', objectFit: 'cover', display: 'block' }} />
                <div style={{ padding: 6, fontSize: 11, lineHeight: 1.4 }}>
                  <div style={{ fontWeight: 600 }}>{it.actual}</div>
                  <div style={{ color: 'var(--text-muted)' }}>{it.name}</div>
                  <div style={{ color: 'var(--text-muted)' }}>similarity {it.similarity.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export function NameView({ initial }: { initial: string }) {
  const [name, setName] = useState(initial);
  const [r, setR] = useState<NameResult | null>(null);
  useEffect(() => {
    if (!name.trim()) return;
    const t = setTimeout(() => { apiPost<NameResult>('/api/classify-name', { name }).then(setR).catch(() => setR(null)); }, 250);
    return () => clearTimeout(t);
  }, [name]);
  return (
    <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
      <input value={name} onChange={(e) => setName(e.target.value)} style={input} aria-label="Product name" />
      {r && (
        <>
          <p style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
            With label words removed, the model reads: “{r.masked_name || '(nothing left)'}”
          </p>
          <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
            {r.models.map((m) => (
              <div key={m.model}>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6 }}>{m.model}</div>
                <Bars top={m.top} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
