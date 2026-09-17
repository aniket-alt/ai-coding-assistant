/**
 * Project 14 client -- multimodal product classification.
 *
 * Every figure comes from /api/results (the training run) or a live API call:
 * product predictions, name classification and photo search.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { GalleryView, NameView, SearchView } from './Explorer';

const TABS: Tab[] = [
  { id: 'results', label: 'Which source wins?' },
  { id: 'leak', label: 'The name leak' },
  { id: 'gallery', label: 'Products' },
  { id: 'search', label: 'Search photos' },
  { id: 'name', label: 'Classify a name' },
  { id: 'classes', label: 'Where fusion helps' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

type Scores = { macro_f1: number; accuracy: number; top3_accuracy: number };
interface Results {
  classes: string[];
  leaks: { label_in_name_pct: number; duplicate_names: number; name_full: Scores; name_masked: Scores;
           random_split_name_model: Scores & { test_names_seen_in_train_pct: number }; grouped_split_name_model: Scores };
  significance: { a: string; b: string; diff: number; ci_low: number; ci_high: number; verdict: string }[];
  best_single: string;
  per_class: { article_type: string; test_products: number; image_f1: number; fusion_f1: number; gain: number }[];
  image_confusions: { actual: string; predicted: string; products: number }[];
  search: { gallery: number; recall_at_10_full_name: number; recall_at_10_masked_name: number; median_rank_full_name: number;
            median_rank_masked_name: number; chance_at_10: number };
  autogluon: { autogluon_version: string; presets: string; time_limit_seconds: number; fit_seconds: number; best_model: string; models_trained: number } | null;
  clip: { model: string; embedding_seconds: number; products: number };
  fusion_reported: string;
}

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)} pts`;

export function App() {
  const [tab, setTab] = useState('results');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const res = useApi<Results>('/api/results');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Contestant" columns={[
      { key: 'macro_f1', label: 'Macro-F1', format: (v) => v.toFixed(4) },
      { key: 'accuracy', label: 'Accuracy', format: (v) => `${(v * 100).toFixed(1)}%` },
      { key: 'top3_accuracy', label: 'Top-3 accuracy', format: (v) => `${(v * 100).toFixed(1)}%` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'results' && (
        <>
          <Panel title="Which source of information wins?"
                 description="Every contestant scored on the same held-out products, grouped so no product shares a name with a training product. Macro-F1 weighs every article type equally.">
            {leaderboard}
          </Panel>
          {res.data && (
            <Panel title="Are the gains real?" description={`Paired bootstrap of the accuracy difference against the best single source that does not read label words: ${res.data.best_single}. A verdict needs the whole 95% interval on one side of zero.`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', minWidth: 680, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                  <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    {['Contestant', 'Accuracy difference', '95% interval', 'Verdict'].map((h, i) => <th key={h} style={{ ...th, textAlign: i === 1 || i === 2 ? 'right' : 'left' }}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {res.data.significance.map((s) => (
                      <tr key={s.a} style={{ borderBottom: '1px solid var(--gridline)' }}>
                        <td style={td}>{s.a}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{signed(s.diff)}</td>
                        <td style={{ ...td, textAlign: 'right' }}>[{signed(s.ci_low)}, {signed(s.ci_high)}]</td>
                        <td style={{ ...td, fontWeight: 600, color: s.verdict === 'better' ? 'var(--delta-up)' : s.verdict === 'worse' ? 'var(--status-critical)' : 'var(--text-secondary)' }}>{s.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </>
      )}

      {tab === 'leak' && res.data && (
        <>
          <Panel title="The name often states the answer"
                 description={`${res.data.leaks.label_in_name_pct.toFixed(1)}% of product names contain a word of their own article type, as in “Men Navy Blue Shirt”. Removing every such word shows how much of the text model is reading the label.`}>
            <StatRow>
              <StatTile label="Name model, full name" value={res.data.leaks.name_full.macro_f1.toFixed(3)} hint="macro-F1" />
              <StatTile label="Name model, label words removed" value={res.data.leaks.name_masked.macro_f1.toFixed(3)} hint="macro-F1" />
            </StatRow>
            <div style={{ marginTop: 'var(--space-4)' }}>
              <Chart kind="bar" height={240}
                     data={[{ model: 'Full name', macro_f1: res.data.leaks.name_full.macro_f1, accuracy: res.data.leaks.name_full.accuracy },
                            { model: 'Label words removed', macro_f1: res.data.leaks.name_masked.macro_f1, accuracy: res.data.leaks.name_masked.accuracy }]}
                     x="model" series={[{ key: 'macro_f1', label: 'Macro-F1' }, { key: 'accuracy', label: 'Accuracy' }]} yFormat={(v) => v.toFixed(2)} />
            </div>
          </Panel>
          <Panel title="Products that share a name"
                 description={`${res.data.leaks.duplicate_names.toLocaleString()} products repeat a display name used by another product. The same name model on a random split, where siblings can sit on both sides, against the grouped split used everywhere else.`}>
            <StatRow>
              <StatTile label="Random split" value={res.data.leaks.random_split_name_model.macro_f1.toFixed(4)}
                        hint={`${res.data.leaks.random_split_name_model.test_names_seen_in_train_pct.toFixed(1)}% of test names also in training`} />
              <StatTile label="Grouped by name" value={res.data.leaks.grouped_split_name_model.macro_f1.toFixed(4)} hint="no test name in training" />
            </StatRow>
          </Panel>
        </>
      )}

      {tab === 'gallery' && res.data && (
        <Panel title="Test products" description="Each product's photo with the image model's and the fusion model's predictions.">
          <GalleryView classes={res.data.classes} />
        </Panel>
      )}

      {tab === 'search' && res.data && (
        <>
          <Panel title="Search photos by description" description={`Your words and every test photo are embedded with ${res.data.clip.model}; the closest photos are shown.`}>
            <SearchView initial="blue denim jeans" />
          </Panel>
          <Panel title="How well names find their own photo" description="For each test product, rank its own photo among all test photos by similarity to its name.">
            <StatRow>
              <StatTile label="Found in top 10, full name" value={`${(res.data.search.recall_at_10_full_name * 100).toFixed(1)}%`} hint={`median rank ${res.data.search.median_rank_full_name.toFixed(0)}`} />
              <StatTile label="Found in top 10, label words removed" value={`${(res.data.search.recall_at_10_masked_name * 100).toFixed(1)}%`} hint={`median rank ${res.data.search.median_rank_masked_name.toFixed(0)}`} />
              <StatTile label="By chance" value={`${(res.data.search.chance_at_10 * 100).toFixed(2)}%`} hint={`${res.data.search.gallery.toLocaleString()} photos`} />
            </StatRow>
          </Panel>
        </>
      )}

      {tab === 'name' && (
        <Panel title="Classify a product name" description="Both text models, live. The second only sees the name after every article-type word has been removed.">
          <NameView initial="Puma Men Running Black Sports Shoes" />
        </Panel>
      )}

      {tab === 'classes' && res.data && (
        <>
          <Panel title="Where fusion helps most" description={`Per-class F1 of ${res.data.fusion_reported}, against the image alone, on test products.`}>
            <Chart kind="bar" horizontal height={res.data.per_class.slice(0, 15).length * 30 + 60} data={res.data.per_class.slice(0, 15)} x="article_type"
                   series={[{ key: 'gain', label: 'F1 gain over image only' }]} yFormat={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
                   caption="The fifteen article types with the largest gains." />
          </Panel>
          <Panel title="What the image model confuses" description="The most frequent mistakes of the image-only model on test products.">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 520, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Actual', 'Predicted', 'Products'].map((h, i) => <th key={h} style={{ ...th, textAlign: i === 2 ? 'right' : 'left' }}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {res.data.image_confusions.map((c) => (
                    <tr key={`${c.actual}|${c.predicted}`} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={td}>{c.actual}</td><td style={td}>{c.predicted}</td><td style={{ ...td, textAlign: 'right' }}>{c.products}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ evaluation: leaderboard || undefined }} />
      )}

      {tab === 'audit' && (
        <>
          <AuditPanel findings={audit.data ?? []}
                      runtimeAssertions={(run?.audit?.runtime_assertions as never) ?? []}
                      summary={{ critical: run?.audit?.critical as number | undefined }} />
          {provenance.data && <ProvenanceBadge provenance={provenance.data} />}
        </>
      )}
    </AppShell>
  );
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
