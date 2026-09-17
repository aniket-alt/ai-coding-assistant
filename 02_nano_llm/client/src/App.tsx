/**
 * Project 02 client -- the language model.
 *
 * Studio, tokenizer and attention views call the live model; training,
 * ablation and check views read the training run.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { AttentionView, GenerationStudio, TokenizerView } from './Studio';

const TABS: Tab[] = [
  { id: 'studio', label: 'Story studio' },
  { id: 'training', label: 'Training & baselines' },
  { id: 'ablations', label: 'Ablations' },
  { id: 'attention', label: 'Attention' },
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'checks', label: 'Checks & samples' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface CurvePoint { step: number; tokens_m: number; train_loss: number; val_loss: number; val_perplexity: number; minutes: number }
interface Training {
  main: { config: Record<string, number | string>; params: number; curve: CurvePoint[]; seconds: number; tokens: number;
          tokens_per_second: number; steps: number; batch: number };
  ngram: { unigram_perplexity: number; bigram_perplexity: number; lambda: number };
  dedup: { train_stories: number; train_duplicate_copies: number; valid_stories: number; valid_in_train: number; valid_scored: number };
  chars_per_token: number; hardware: string; torch: string;
}
interface Ablation {
  variant: string; seed: number; params: number; val_loss: number; val_perplexity: number; tokens_per_second: number;
  loss_vs_reference: number; beyond_seed_noise: boolean;
}
interface Ablations { ablations: Ablation[]; seed_gap: number; steps: number }
interface Checks {
  checks: { causal_max_logit_change: number; cache_identical: number; cache_prompts: number;
            timings: { cache: boolean; tokens: number; ms_per_token: number }[] };
  samples: { prompt: string; continuation: string }[];
}

export function App() {
  const [tab, setTab] = useState('studio');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const training = useApi<Training>('/api/training');
  const abl = useApi<Ablations>('/api/ablations');
  const checks = useApi<Checks>('/api/checks');
  const run = metrics.data;
  const prompts = checks.data?.samples.map((s) => s.prompt) ?? [];

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Model" showTiming columns={[
      { key: 'val_perplexity', label: 'Validation perplexity', format: (v) => v.toFixed(2) },
      { key: 'val_loss', label: 'Loss (nats/token)', format: (v) => v.toFixed(4) },
      { key: 'bits_per_char', label: 'Bits per character', format: (v) => v.toFixed(3) },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'studio' && (
        <Panel title="Story studio" description="The trained model continues your prompt. Everything is generated live by the served checkpoint.">
          {checks.data ? <GenerationStudio prompts={prompts} /> : <p style={{ color: 'var(--text-muted)' }}>{checks.error ?? 'Loading…'}</p>}
        </Panel>
      )}

      {tab === 'training' && training.data && (
        <>
          <Panel title="How good is it, compared with what?"
                 description="Perplexity on validation stories the model never saw. Lower is better; a perplexity of p means the model is as uncertain as a fair choice among p tokens.">
            {leaderboard}
          </Panel>
          <Panel title="Learning curve"
                 description={`${training.data.main.params.toLocaleString()} parameters, ${training.data.main.steps.toLocaleString()} steps on ${training.data.hardware}. Validation loss is measured on a fixed sample of windows during training.`}>
            <StatRow>
              <StatTile label="Tokens seen" value={(training.data.main.tokens / 1e6).toFixed(0)} unit="M" />
              <StatTile label="Training time" value={(training.data.main.seconds / 60).toFixed(0)} unit="min" />
              <StatTile label="Throughput" value={(training.data.main.tokens_per_second / 1e3).toFixed(0)} unit="k tokens/s" />
            </StatRow>
            <div style={{ marginTop: 'var(--space-4)' }}>
              <Chart kind="line" height={320} data={training.data.main.curve} x="tokens_m" xLabel="Training tokens (millions)"
                     xFormat={(v) => Number(v).toFixed(0)}
                     series={[{ key: 'train_loss', label: 'Training loss' }, { key: 'val_loss', label: 'Validation loss' }]}
                     yFormat={(v) => v.toFixed(2)} />
            </div>
          </Panel>
          <Panel title="Keeping the score honest"
                 description="A validation story that also appears in training would flatter the score, so every story in both files is hashed and any overlap removed before scoring.">
            <StatRow>
              <StatTile label="Training stories" value={training.data.dedup.train_stories.toLocaleString()}
                        hint={training.data.dedup.train_duplicate_copies === 1 ? '1 is an exact repeat of an earlier story'
                          : `${training.data.dedup.train_duplicate_copies.toLocaleString()} are exact repeats of an earlier story`} />
              <StatTile label="Validation stories seen in training" value={training.data.dedup.valid_in_train.toLocaleString()}
                        hint={`of ${training.data.dedup.valid_stories.toLocaleString()}${training.data.dedup.valid_in_train ? '; removed before scoring' : ''}`} />
              <StatTile label="Validation stories scored" value={training.data.dedup.valid_scored.toLocaleString()} />
            </StatRow>
          </Panel>
        </>
      )}

      {tab === 'ablations' && abl.data && (
        <Panel title="Which modern components earn their place?"
               description={`Each variant trains for ${abl.data.steps.toLocaleString()} steps with one component swapped for its older equivalent. The reference was trained twice with different seeds; that gap (${abl.data.seed_gap.toFixed(4)} nats/token) is the noise floor.`}>
          <Chart kind="bar" horizontal height={300} data={abl.data.ablations} x="variant"
                 series={[{ key: 'loss_vs_reference', label: 'Validation loss vs reference (nats/token)' }]}
                 yFormat={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`}
                 caption="Positive means worse than the reference. A difference smaller than the seed gap is indistinguishable from noise." />
          <div style={{ overflowX: 'auto', marginTop: 'var(--space-4)' }}>
            <table style={{ width: '100%', minWidth: 720, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
              <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                {['Variant', 'Parameters', 'Perplexity', 'Loss vs reference', 'Beyond seed noise?', 'Tokens/s'].map((h, i) => (
                  <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
              </tr></thead>
              <tbody>
                {abl.data.ablations.map((a) => (
                  <tr key={a.variant} style={{ borderBottom: '1px solid var(--gridline)' }}>
                    <td style={td}>{a.variant}</td>
                    <td style={{ ...td, textAlign: 'right' }}>{a.params.toLocaleString()}</td>
                    <td style={{ ...td, textAlign: 'right' }}>{a.val_perplexity.toFixed(2)}</td>
                    <td style={{ ...td, textAlign: 'right' }}>{`${a.loss_vs_reference >= 0 ? '+' : ''}${a.loss_vs_reference.toFixed(4)}`}</td>
                    <td style={{ ...td, textAlign: 'right' }}>{a.variant.startsWith('Reference') ? '—' : a.beyond_seed_noise ? 'yes' : 'no'}</td>
                    <td style={{ ...td, textAlign: 'right' }}>{Math.round(a.tokens_per_second).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {tab === 'attention' && training.data && (
        <Panel title="Where each token looks" description="Attention weights from the served model, per layer and head.">
          <AttentionView initial="Lily found a red ball. She gave the ball to her friend Tom."
                         heads={Number(training.data.main.config.n_heads)} layers={Number(training.data.main.config.n_layers)} />
        </Panel>
      )}

      {tab === 'tokenizer' && (
        <Panel title="How the model reads text" description="A byte-level BPE tokenizer trained on the training stories.">
          <TokenizerView initial="Once upon a time, a curious little fox discovered a sparkling river." />
        </Panel>
      )}

      {tab === 'checks' && checks.data && (
        <>
          <Panel title="Mechanics checked by execution" description="Run during training and recorded in the audit trail.">
            <StatRow>
              <StatTile label="Causal mask" value={checks.data.checks.causal_max_logit_change.toExponential(1)}
                        hint="Largest change in earlier logits when later tokens were replaced" />
              <StatTile label="KV cache agrees" value={`${checks.data.checks.cache_identical} / ${checks.data.checks.cache_prompts}`}
                        hint="Greedy continuations identical with and without the cache" />
              {checks.data.checks.timings.map((t) => (
                <StatTile key={String(t.cache)} label={t.cache ? 'With KV cache' : 'Without KV cache'}
                          value={t.ms_per_token.toFixed(1)} unit="ms / token" hint={`${t.tokens} tokens`} />
              ))}
            </StatRow>
          </Panel>
          <Panel title="Fixed-seed samples" description="Temperature 0.8, top-p 0.9, repetition penalty 1.1, recorded at the end of training.">
            {checks.data.samples.map((s) => (
              <p key={s.prompt} style={{ lineHeight: 1.7, borderBottom: '1px solid var(--gridline)', paddingBottom: 'var(--space-3)' }}>
                <span style={{ color: 'var(--text-muted)' }}>{s.prompt}</span>{s.continuation}
              </p>
            ))}
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
