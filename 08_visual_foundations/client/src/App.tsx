/**
 * Project 08 client -- visual foundations.
 *
 * Four lessons on real data, a quiz, and the shared CRISP-DM and audit views.
 * Every figure comes from the API: live computations by the checked
 * implementations, or the recorded training run.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, ProvenanceBadge,
  StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { BackpropLesson, BayesLesson, DescentLesson, ThresholdLesson } from './Lessons';
import { Quiz } from './Quiz';

const TABS: Tab[] = [
  { id: 'bayes', label: '1 · Probability' },
  { id: 'evaluation', label: '2 · Evaluation' },
  { id: 'descent', label: '3 · Gradient descent' },
  { id: 'backprop', label: '4 · Backpropagation' },
  { id: 'quiz', label: '5 · Quiz' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

const EXAMPLES = [
  'URGENT! You have won a 1 week FREE membership. Text CLAIM to 81010 now',
  'Are we still meeting for lunch tomorrow? Let me know',
  'Your account has been credited. Call now to collect your cash prize',
];

type Stat = { mean: number; sd: number };
interface Spam {
  sms: { messages: number; spam_pct: number; duplicates: number; duplicate_spam_share_pct: number; train: number; test: number;
         vocabulary: number; sklearn_gap: number;
         top_tokens: { spam: { token: string; log_ratio: number }[]; ham: { token: string; log_ratio: number }[] } };
  leak: Record<string, { average_precision: Stat; f1: Stat; recall: Stat; seen_in_train_pct: Stat }>;
  leak_seeds: number;
  pr_curve: { threshold: number; precision: number; recall: number }[];
}
interface Gd {
  features: Record<string, string>; exact_test_rmse: number; mean_baseline_test_rmse: number;
  stable_rate_scaled: number; stable_rate_raw: number; condition_scaled: number; condition_raw: number;
  scaled_after_500: number; raw_after_500: number; exact_mse: number;
  runs: { scaled: boolean; lr: number; diverged: boolean; final_mse: number | null }[];
}
interface Backprop {
  hidden: number; epochs: number; test_rmse: number; linear_test_rmse: number;
  losses: { epoch: number; mse_standardised: number }[];
  gradient_check: { before: { parameters_checked: number; max_relative_error: number }; after: { max_relative_error: number } };
}

export function App() {
  const [tab, setTab] = useState('bayes');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const spam = useApi<Spam>('/api/spam');
  const gd = useApi<Gd>('/api/gradient-descent');
  const bp = useApi<Backprop>('/api/backprop/training');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Spam filter" columns={[
      { key: 'average_precision', label: 'Average precision', format: (v) => v.toFixed(4) },
      { key: 'f1', label: 'F1 at 0.5', format: (v) => v.toFixed(3) },
      { key: 'precision', label: 'Precision', format: (v) => `${(v * 100).toFixed(1)}%` },
      { key: 'recall', label: 'Recall', format: (v) => `${(v * 100).toFixed(1)}%` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'bayes' && (
        <>
          <Panel title="How Naive Bayes decides" description="Type any message. The written-out classifier, matched against scikit-learn during training, weighs each word's evidence.">
            <BayesLesson examples={EXAMPLES} />
          </Panel>
          {spam.data && (
            <Panel title="The strongest evidence it learned" description={`From ${spam.data.sms.train.toLocaleString()} training messages and a vocabulary of ${spam.data.sms.vocabulary.toLocaleString()} words.`}>
              <div style={{ display: 'grid', gap: 'var(--space-4)', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
                <Chart kind="bar" horizontal height={440} data={spam.data.sms.top_tokens.spam} x="token"
                       series={[{ key: 'log_ratio', label: 'Most spam-like words' }]} yFormat={(v) => v.toFixed(1)} />
                <Chart kind="bar" horizontal height={440} data={spam.data.sms.top_tokens.ham} x="token"
                       series={[{ key: 'log_ratio', label: 'Most ham-like words', color: 'var(--series-2)' }]} yFormat={(v) => v.toFixed(1)} />
              </div>
            </Panel>
          )}
        </>
      )}

      {tab === 'evaluation' && (
        <>
          <Panel title="Spam filters, compared" description="On held-out messages, after removing duplicates so no message is tested on after being trained on.">
            {leaderboard}
          </Panel>
          <Panel title="Moving the threshold" description="Every threshold trades missed spam against wrongly flagged messages.">
            <ThresholdLesson />
          </Panel>
          {spam.data && (
            <>
              <Panel title="Precision against recall" description="Naive Bayes on the held-out messages, across every threshold.">
                <Chart kind="scatter" height={300} data={spam.data.pr_curve} x="recall" xLabel="Recall"
                       series={[{ key: 'precision', label: 'Precision' }]} yFormat={(v) => v.toFixed(2)}
                       caption="Each point is one threshold: recall on the horizontal axis, precision on the vertical." />
              </Panel>
              <Panel title="What duplicated messages do to a split"
                     description={`${spam.data.sms.duplicates.toLocaleString()} messages are exact copies of earlier ones. Naive Bayes over ${spam.data.leak_seeds} random splits, mean ± standard deviation.`}>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', minWidth: 620, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                    <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                      {['Split', 'Test messages already in training', 'Average precision', 'F1', 'Recall'].map((h, i) => (
                        <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                    </tr></thead>
                    <tbody>
                      {Object.entries(spam.data.leak).map(([mode, s]) => (
                        <tr key={mode} style={{ borderBottom: '1px solid var(--gridline)' }}>
                          <td style={td}>{mode}</td>
                          <td style={{ ...td, textAlign: 'right' }}>{s.seen_in_train_pct.mean.toFixed(1)}%</td>
                          <td style={{ ...td, textAlign: 'right' }}>{s.average_precision.mean.toFixed(3)} ± {s.average_precision.sd.toFixed(3)}</td>
                          <td style={{ ...td, textAlign: 'right' }}>{s.f1.mean.toFixed(3)} ± {s.f1.sd.toFixed(3)}</td>
                          <td style={{ ...td, textAlign: 'right' }}>{s.recall.mean.toFixed(3)} ± {s.recall.sd.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          )}
        </>
      )}

      {tab === 'descent' && (
        <>
          <Panel title="Gradient descent, live" description="Linear regression of a power plant's output on four weather readings, run on the server as you change the settings.">
            <DescentLesson />
          </Panel>
          {gd.data && (
            <Panel title="Why scaling matters here" description="Recorded during training on the same readings.">
              <StatRow>
                <StatTile label="Condition number, scaled" value={gd.data.condition_scaled.toFixed(1)} />
                <StatTile label="Condition number, raw" value={Math.round(gd.data.condition_raw).toLocaleString()} />
                <StatTile label="MSE after 500 steps, scaled" value={gd.data.scaled_after_500.toFixed(2)} hint="learning rate 0.1" />
                <StatTile label="MSE after 500 steps, raw" value={gd.data.raw_after_500.toFixed(1)} hint="just under the raw stable limit" />
              </StatRow>
              <div style={{ overflowX: 'auto', marginTop: 'var(--space-4)' }}>
                <table style={{ width: '100%', minWidth: 520, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                  <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    {['Features', 'Learning rate', 'After 500 steps'].map((h, i) => <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {gd.data.runs.map((r) => (
                      <tr key={`${r.scaled}${r.lr}`} style={{ borderBottom: '1px solid var(--gridline)' }}>
                        <td style={td}>{r.scaled ? 'scaled' : 'raw'}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{r.lr < 0.001 ? r.lr.toExponential(3) : r.lr}</td>
                        <td style={{ ...td, textAlign: 'right', color: r.diverged ? 'var(--status-critical)' : undefined }}>
                          {r.diverged ? 'diverged' : `MSE ${(r.final_mse ?? 0).toFixed(2)}`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </>
      )}

      {tab === 'backprop' && (
        <>
          <Panel title="Backpropagation, one row at a time" description="The trained 4–8–1 network on one power-plant reading: forward values in the nodes, gradients beside them.">
            <BackpropLesson />
          </Panel>
          {bp.data && (
            <Panel title="Training and checking the network" description={`${bp.data.epochs.toLocaleString()} full-batch gradient steps.`}>
              <StatRow>
                <StatTile label="Test RMSE, network" value={bp.data.test_rmse.toFixed(2)} unit="MW" />
                <StatTile label="Test RMSE, linear least squares" value={bp.data.linear_test_rmse.toFixed(2)} unit="MW" />
                <StatTile label="Gradient check" value={bp.data.gradient_check.after.max_relative_error.toExponential(1)}
                          hint={`largest relative error over ${bp.data.gradient_check.before.parameters_checked} parameters, after training`} />
              </StatRow>
              <div style={{ marginTop: 'var(--space-4)' }}>
                <Chart kind="line" height={240} data={bp.data.losses} x="epoch" xLabel="Step"
                       series={[{ key: 'mse_standardised', label: 'Training MSE (standardised output)' }]}
                       yFormat={(v) => v.toFixed(3)} zeroBaseline={false} />
              </div>
            </Panel>
          )}
        </>
      )}

      {tab === 'quiz' && (
        <Panel title="Check your understanding" description="Answers are checked on the server; each explanation quotes this run's results.">
          <Quiz />
        </Panel>
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
