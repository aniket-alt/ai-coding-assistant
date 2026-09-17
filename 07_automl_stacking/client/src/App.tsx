/**
 * Project 07 client -- the AutoML tournament.
 *
 * Every figure comes from the training run or the prediction API. The only
 * project-specific pieces are the call form's field list and the tables of
 * bootstrap verdicts and AutoGluon's own leaderboard.
 */

import { useState } from 'react';
import {
  AppShell, AuditPanel, Chart, CrispDmWalkthrough, ModelLeaderboard, Panel, PredictionForm,
  ProvenanceBadge, StatRow, StatTile, useApi, useAudit, useCrispDm, useMeta, useMetrics, useProvenance,
} from '@dsfs/ui';
import type { FieldSpec, Tab } from '@dsfs/ui';

const TABS: Tab[] = [
  { id: 'tournament', label: 'Tournament' },
  { id: 'predict', label: 'Score a client' },
  { id: 'stacking', label: 'Inside the stack' },
  { id: 'shift', label: 'Shift & leak' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'audit', label: 'Audit' },
];

interface Form {
  choices: Record<string, string[]>;
  defaults: Record<string, string | number>;
  ranges: Record<string, [number, number]>;
}
interface Member { model: string; probability: number; weight: number; contribution: number }
interface Prediction {
  probability: number; tuned_lightgbm: number; stack: number; members: Member[];
  train_rate_pct: number; test_rate_pct: number; served: string;
}
interface Verdict { a: string; b: string; diff: number; ci_low: number; ci_high: number; verdict: string }
interface EstimateCheck {
  random_cv_auc: number; time_ordered_auc: number; actual_test_auc: number;
  time_ordered_train: string; time_ordered_validate: string;
}
interface Leak { honest_test_auc: number; leaky_test_auc: number; honest_believed_auc: number; leaky_believed_auc: number }
interface AutoGluon {
  autogluon_version: string; presets: string; time_limit_seconds: number; fit_seconds: number;
  best_model: string; believed_validation_auc: number; models_trained: number;
  leaderboard: { model: string; score_val: number; stack_level: number; fit_time: number }[];
}
interface Tournament { significance: Verdict[]; estimate_check: EstimateCheck; autogluon: AutoGluon | null; leak: Leak }
interface Stacking {
  level_one: { model: string; oof_auc: number; test_auc: number; fit_seconds: number; weight: number; stacker_coef: number }[];
  trajectory: { round: number; added: string; oof_auc: number }[];
  tuning_trace: { trial: number; cv_auc: number; best_so_far: number }[];
  tuned_params: Record<string, number>;
}
interface Shift {
  shift: {
    periods: { period: string; calls: number; subscribed_pct: number; euribor3m: number; is_test: boolean }[];
    train_rate_pct: number; test_rate_pct: number; train_calls: number; test_calls: number;
    pdays_never_pct: number;
  };
  leak: Leak;
  estimate_check: EstimateCheck;
  importance: { feature: string; auc_drop: number; spread: number; kind: string }[];
}

const LABELS: Record<string, string> = {
  age: 'Age', campaign: 'Calls this campaign', pdays: 'Days since last campaign (999 = never)',
  previous: 'Contacts before this campaign', 'emp.var.rate': 'Employment variation rate',
  'cons.price.idx': 'Consumer price index', 'cons.conf.idx': 'Consumer confidence index',
  euribor3m: 'Euribor 3-month rate', 'nr.employed': 'Employees (thousands)', job: 'Job', marital: 'Marital status',
  education: 'Education', default: 'Credit in default', housing: 'Housing loan', loan: 'Personal loan',
  contact: 'Contact type', month: 'Month of call', day_of_week: 'Day of call', poutcome: 'Previous campaign outcome',
};

function fieldsFor(form: Form): FieldSpec[] {
  const numeric = Object.keys(form.ranges).map((name): FieldSpec => ({
    kind: 'number', name, label: LABELS[name] ?? name, min: form.ranges[name][0], max: form.ranges[name][1],
    step: ['age', 'campaign', 'pdays', 'previous'].includes(name) ? 1 : 0.001,
  }));
  const categorical = Object.keys(form.choices).map((name): FieldSpec => ({
    kind: 'select', name, label: LABELS[name] ?? name,
    options: form.choices[name].map((v) => ({ value: v, label: v })),
  }));
  return [...categorical, ...numeric];
}

export function App() {
  const [tab, setTab] = useState('tournament');
  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const form = useApi<Form>('/api/form');
  const tour = useApi<Tournament>('/api/tournament');
  const stack = useApi<Stacking>('/api/stacking');
  const shift = useApi<Shift>('/api/shift');
  const run = metrics.data;

  const leaderboard = run && (
    <ModelLeaderboard run={run} entityLabel="Contestant" showTiming columns={[
      { key: 'roc_auc', label: 'ROC AUC, 2010', format: (v) => v.toFixed(4) },
      { key: 'average_precision', label: 'Avg precision', format: (v) => v.toFixed(4) },
      { key: 'believed_auc', label: 'Believed AUC', format: (v) => v.toFixed(4) },
      { key: 'optimism', label: 'Overestimate', goodWhenUp: false, format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}` },
    ]} />
  );

  return (
    <AppShell meta={meta.data} provenance={provenance.data} tabs={TABS} active={tab}
              onTabChange={setTab} repoUrl="https://github.com/aniket-alt/ai-coding-assistant">
      {tab === 'tournament' && (
        <>
          <Panel title="Which contestant ranks 2010's clients best?"
                 description="Trained on calls before 2010, scored on every call in 2010. Believed AUC is what each method's own validation promised.">
            {leaderboard}
          </Panel>
          {run && (
            <Panel title="What each method promised, and what it delivered"
                   description="Believed AUC from the method's own validation on pre-2010 calls, beside its AUC on 2010 calls.">
              <Chart kind="bar" height={320}
                     data={run.models.filter((m) => m.metrics.believed_auc !== undefined && m.family !== 'baseline')
                       .map((m) => ({ model: m.name, believed: m.metrics.believed_auc, actual: m.metrics.roc_auc }))}
                     x="model" series={[{ key: 'believed', label: 'Believed AUC' }, { key: 'actual', label: 'Actual AUC in 2010' }]}
                     yFormat={(v) => v.toFixed(2)}
                     caption="Every gap is the distance between a random validation split and the year the model was deployed into." />
            </Panel>
          )}
          {tour.data && (
            <Panel title="Are the differences real?"
                   description="Paired bootstrap over the same resampled 2010 calls, 2,000 resamples. A verdict needs the whole 95% interval on one side of zero.">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', minWidth: 720, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                  <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    {['Contestant', 'Compared with', 'AUC difference', '95% interval', 'Verdict'].map((h, i) => (
                      <th key={h} style={{ ...th, textAlign: i >= 2 && i <= 3 ? 'right' : 'left' }}>{h}</th>))}
                  </tr></thead>
                  <tbody>
                    {tour.data.significance.map((s) => (
                      <tr key={`${s.a}|${s.b}`} style={{ borderBottom: '1px solid var(--gridline)' }}>
                        <td style={td}>{s.a}</td>
                        <td style={td}>{s.b}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{signed(s.diff, 4)}</td>
                        <td style={{ ...td, textAlign: 'right' }}>[{signed(s.ci_low, 4)}, {signed(s.ci_high, 4)}]</td>
                        <td style={{ ...td, fontWeight: 600, color: s.verdict === 'better' ? 'var(--delta-up)'
                          : s.verdict === 'worse' ? 'var(--status-critical)' : 'var(--text-secondary)' }}>{s.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          {tour.data?.autogluon && (
            <Panel title="AutoGluon's own leaderboard"
                   description={`AutoGluon ${tour.data.autogluon.autogluon_version}, ${tour.data.autogluon.presets} presets, ${tour.data.autogluon.time_limit_seconds}s limit: ${tour.data.autogluon.models_trained} models in ${tour.data.autogluon.fit_seconds.toFixed(0)}s. Scores are its internal validation, not 2010.`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', minWidth: 560, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                  <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    {['Model', 'Stack level', 'Internal validation AUC', 'Fit (s)'].map((h, i) => (
                      <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                  </tr></thead>
                  <tbody>
                    {tour.data.autogluon.leaderboard.map((m) => (
                      <tr key={m.model} style={{ borderBottom: '1px solid var(--gridline)' }}>
                        <td style={{ ...td, fontWeight: m.model === tour.data!.autogluon!.best_model ? 700 : 400 }}>{m.model}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{m.stack_level}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{m.score_val.toFixed(4)}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{m.fit_time.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
        </>
      )}

      {tab === 'predict' && (
        <Panel title="Score a client before calling"
               description="The served model's chance that this client subscribes, with the hand-built stack's answer broken into its members.">
          {form.data ? (
            <PredictionForm fields={fieldsFor(form.data)} endpoint="/api/predict"
              initialValues={form.data.defaults} autoSubmit
              renderResult={(r, { loading }) => {
                const p = r as Prediction | null;
                if (!p) return <p style={{ color: 'var(--text-muted)' }}>{loading ? 'Scoring…' : ''}</p>;
                return (
                  <div style={{ opacity: loading ? 0.55 : 1 }}>
                    <StatRow>
                      <StatTile label="Chance of subscribing" value={(p.probability * 100).toFixed(1)} unit="%" size="hero"
                                accent="var(--series-1)" hint={`Served: ${p.served}`} />
                      <StatTile label="Tuned LightGBM" value={(p.tuned_lightgbm * 100).toFixed(1)} unit="%" />
                      <StatTile label="Hand-built stack" value={(p.stack * 100).toFixed(1)} unit="%" />
                    </StatRow>
                    <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                      For context, {p.train_rate_pct.toFixed(1)}% of calls before 2010 led to a subscription,
                      and {p.test_rate_pct.toFixed(1)}% of calls in 2010.
                    </p>
                    <Chart kind="bar" horizontal height={200} data={p.members} x="model"
                           series={[{ key: 'probability', label: "Member's own score" }, { key: 'contribution', label: 'Weighted contribution' }]}
                           yFormat={(v) => v.toFixed(2)}
                           caption="The stack's answer is the sum of the weighted contributions. Weights were chosen by Caruana selection." />
                  </div>
                );
              }} />
          ) : <p style={{ color: 'var(--text-muted)' }}>{form.error ?? 'Loading…'}</p>}
        </Panel>
      )}

      {tab === 'stacking' && stack.data && (
        <>
          <Panel title="Level one: five bagged models"
                 description="Each model is trained on five folds of pre-2010 calls. Out-of-fold AUC is what stacking sees; 2010 AUC is what happened.">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', minWidth: 720, borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead><tr style={{ borderBottom: '2px solid var(--axis)' }}>
                  {['Model', 'Out-of-fold AUC', '2010 AUC', 'Caruana weight', 'Stacker coefficient', 'Fit (s)'].map((h, i) => (
                    <th key={h} style={{ ...th, textAlign: i ? 'right' : 'left' }}>{h}</th>))}
                </tr></thead>
                <tbody>
                  {stack.data.level_one.map((m) => (
                    <tr key={m.model} style={{ borderBottom: '1px solid var(--gridline)' }}>
                      <td style={td}>{m.model}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{m.oof_auc.toFixed(4)}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{m.test_auc.toFixed(4)}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{m.weight ? `${(m.weight * 100).toFixed(0)}%` : '—'}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{signed(m.stacker_coef, 3)}</td>
                      <td style={{ ...td, textAlign: 'right' }}>{m.fit_seconds.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Caruana selection, round by round"
                 description="Each round adds the model that most improves the out-of-fold AUC of the running average. Repeats raise a model's weight.">
            <Chart kind="line" height={260} data={stack.data.trajectory} x="round" xLabel="Round"
                   series={[{ key: 'oof_auc', label: 'Out-of-fold AUC of the ensemble' }]} yFormat={(v) => v.toFixed(4)} zeroBaseline={false}
                   caption={`Models added, in order: ${stack.data.trajectory.map((t) => t.added).join(' → ')}`} />
          </Panel>
          <Panel title="Tuning LightGBM" description="Optuna trials, each scored by five-fold cross-validation on pre-2010 calls.">
            <Chart kind="line" height={260} data={stack.data.tuning_trace} x="trial" xLabel="Trial"
                   series={[{ key: 'cv_auc', label: 'Trial CV AUC' }, { key: 'best_so_far', label: 'Best so far' }]}
                   yFormat={(v) => v.toFixed(3)} zeroBaseline={false} />
          </Panel>
        </>
      )}

      {tab === 'shift' && shift.data && (
        <>
          <Panel title="The year the model is deployed into is not the years it learned from"
                 description="Share of calls ending in a subscription, by month. The file stores only the month; years are reconstructed from its date order.">
            <StatRow>
              <StatTile label="Subscribed, before 2010" value={shift.data.shift.train_rate_pct.toFixed(1)} unit="%"
                        hint={`${shift.data.shift.train_calls.toLocaleString()} training calls`} />
              <StatTile label="Subscribed, 2010" value={shift.data.shift.test_rate_pct.toFixed(1)} unit="%"
                        hint={`${shift.data.shift.test_calls.toLocaleString()} test calls`} />
            </StatRow>
            <div style={{ marginTop: 'var(--space-4)' }}>
              <Chart kind="bar" height={280} data={shift.data.shift.periods} x="period"
                     series={[{ key: 'subscribed_pct', label: 'Calls ending in a subscription (%)' }]}
                     yFormat={(v) => `${v.toFixed(0)}%`} />
              <Chart kind="line" height={220} data={shift.data.shift.periods} x="period"
                     series={[{ key: 'euribor3m', label: 'Euribor 3-month rate (%)' }]} yFormat={(v) => v.toFixed(1)}
                     caption="Months from 2010 onward form the test set. The interest rate collapsed during the financial crisis." />
            </div>
          </Panel>
          <Panel title="Three estimates of one model's 2010 score"
                 description={`The same default LightGBM. Random cross-validation was too ${direction(shift.data.estimate_check.random_cv_auc, shift.data.estimate_check.actual_test_auc)}; validating on the following year was too ${direction(shift.data.estimate_check.time_ordered_auc, shift.data.estimate_check.actual_test_auc)}.`}>
            <StatRow>
              <StatTile label="Random five-fold CV" value={shift.data.estimate_check.random_cv_auc.toFixed(4)} hint="Pre-2010 calls, shuffled" />
              <StatTile label="Time-ordered check" value={shift.data.estimate_check.time_ordered_auc.toFixed(4)}
                        hint={`Train ${shift.data.estimate_check.time_ordered_train}, validate ${shift.data.estimate_check.time_ordered_validate}`} />
              <StatTile label="Actual, 2010" value={shift.data.estimate_check.actual_test_auc.toFixed(4)} />
            </StatRow>
          </Panel>
          <Panel title="The duration leak, measured"
                 description="`duration` is the length of the call, known only once the client has answered. Default LightGBM with and without it.">
            <StatRow>
              <StatTile label="2010 AUC, honest" value={shift.data.leak.honest_test_auc.toFixed(4)} />
              <StatTile label="2010 AUC, with duration" value={shift.data.leak.leaky_test_auc.toFixed(4)}
                        hint={`Believed ${shift.data.leak.leaky_believed_auc.toFixed(4)} in cross-validation`} />
            </StatRow>
          </Panel>
          <Panel title="What the stack relies on in 2010"
                 description="Drop in the stack's 2010 AUC when one feature is shuffled, averaged over three shuffles.">
            <Chart kind="bar" horizontal height={shift.data.importance.length * 30 + 60} data={shift.data.importance} x="feature"
                   series={[{ key: 'auc_drop', label: 'AUC lost when shuffled' }]} yFormat={(v) => v.toFixed(3)}
                   caption="A negative value means the stack scored better on 2010 calls with that feature scrambled: what it learned from the feature before 2010 did not hold in 2010." />
          </Panel>
        </>
      )}

      {tab === 'crispdm' && crispdm.data && (
        <CrispDmWalkthrough report={crispdm.data} slots={{ modeling: leaderboard || undefined }} />
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

/** Whether an estimate sat above or below what happened. */
function direction(estimate: number, actual: number): string {
  return estimate > actual ? 'optimistic' : 'pessimistic';
}

function signed(v: number, digits: number): string {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`;
}

const th: React.CSSProperties = {
  padding: '8px 10px', color: 'var(--text-muted)', fontWeight: 600, fontSize: 'var(--text-xs)',
  textTransform: 'uppercase', letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = { padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap' };
