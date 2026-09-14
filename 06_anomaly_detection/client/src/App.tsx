/**
 * Project 06 client.
 *
 * The only project-specific view is TransactionScorer. Metrics, the detector
 * tournament, the CRISP-DM walkthrough, the audit and every chart come from
 * @dsfs/ui unchanged -- which is the point of this project being second: it
 * tests whether components designed against a regression problem survive
 * contact with ranking under extreme imbalance.
 */

import { useState } from 'react';
import {
  AppShell,
  AuditPanel,
  Chart,
  CrispDmWalkthrough,
  MetricsPanel,
  ModelLeaderboard,
  Panel,
  ProvenanceBadge,
  useApi,
  useAudit,
  useCrispDm,
  useMeta,
  useMetrics,
  useProvenance,
} from '@dsfs/ui';
import type { Tab } from '@dsfs/ui';
import { TransactionScorer } from './TransactionScorer';

const TABS: Tab[] = [
  { id: 'scorer', label: 'Transaction scorer' },
  { id: 'budget', label: 'Review budget' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'models', label: 'Detector tournament' },
  { id: 'audit', label: 'Audit' },
];

interface BudgetPoint {
  alerts: number;
  caught: number;
  precision_pct: number;
  recall_pct: number;
}

interface Eda {
  hourly_profile: { hour: number; transactions: number; fraud: number; fraud_rate_pct: number }[];
  amount_bands: { band: string; transactions: number; fraud: number; fraud_rate_pct: number }[];
}

export function App() {
  const [tab, setTab] = useState('scorer');

  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const budget = useApi<BudgetPoint[]>('/api/budget-curve');
  const eda = useApi<Eda>('/api/eda');

  const run = metrics.data;

  return (
    <AppShell
      meta={meta.data}
      provenance={provenance.data}
      tabs={TABS}
      active={tab}
      onTabChange={setTab}
      repoUrl="https://github.com/aniket-alt/ai-coding-assistant"
    >
      {metrics.notBuilt && (
        <Notice>
          No training artifacts yet. Run <code>python ml/train.py</code>, then reload.
        </Notice>
      )}

      {tab === 'scorer' && (
        <>
          <Panel
            title="Score a transaction"
            description="Real held-out transactions, with the true label shown so misses are visible."
          >
            <TransactionScorer />
          </Panel>

          {run && (
            <Panel
              title="How the detector performs"
              description="Ranked on average precision, because accuracy cannot tell these models apart."
            >
              <MetricsPanel
                run={run}
                highlight={[
                  { key: 'avg_precision', label: 'Average precision',
                    format: (v) => v.toFixed(4),
                    hint: 'Area under precision-recall. Sensitive to the rare class.' },
                  { key: 'recall_at_budget', label: 'Fraud caught', unit: '%',
                    format: (v) => (v * 100).toFixed(1),
                    hint: 'Within a 100-alert daily review budget.' },
                  { key: 'roc_auc', label: 'ROC-AUC', format: (v) => v.toFixed(4) },
                  { key: 'accuracy', label: 'Accuracy', unit: '%',
                    format: (v) => (v * 100).toFixed(3),
                    hint: 'Useless here: predicting nothing scores 99.8%.' },
                ]}
              />
            </Panel>
          )}
        </>
      )}

      {tab === 'budget' && (
        <>
          <Panel
            title="What an analyst team can actually catch"
            description="Precision and recall at each review budget, measured on the held-out window."
          >
            {budget.data ? (
              <Chart
                kind="line"
                height={320}
                data={budget.data}
                x="alerts"
                series={[
                  { key: 'precision_pct', label: 'Precision (%)' },
                  { key: 'recall_pct', label: 'Fraud caught (%)' },
                ]}
                xLabel="Alerts reviewed per day"
                yFormat={(v) => `${v.toFixed(0)}%`}
                caption="The trade every fraud team makes: reviewing more transactions catches more fraud, and wastes more analyst time. Both curves are measured, not modelled."
              />
            ) : (
              <Notice>{budget.error ?? 'Loading the budget curve…'}</Notice>
            )}
          </Panel>

          {eda.data && (
            <>
              <Panel
                title="Fraud rate by transaction size"
                description="The assumption that fraud means large transactions does not survive the data."
              >
                <Chart
                  kind="bar"
                  height={260}
                  data={eda.data.amount_bands}
                  x="band"
                  series={[{ key: 'fraud_rate_pct', label: 'Fraud rate (%)' }]}
                  yFormat={(v) => `${v.toFixed(2)}%`}
                  caption="Median fraudulent amount is €9.25 against €22.00 legitimate — consistent with card-testing, where a stolen number is validated with a trivial charge first."
                />
              </Panel>

              <Panel
                title="Fraud rate across the 48-hour window"
                description="Fraud arrives in campaigns, which is why the split is chronological."
              >
                <Chart
                  kind="line"
                  height={260}
                  data={eda.data.hourly_profile}
                  x="hour"
                  series={[{ key: 'fraud_rate_pct', label: 'Fraud rate (%)' }]}
                  xLabel="Hour of window"
                  yFormat={(v) => `${v.toFixed(3)}%`}
                  caption="Clustering in time is exactly why a random split would flatter every model: it would place the same campaign on both sides of the boundary."
                />
              </Panel>
            </>
          )}
        </>
      )}

      {tab === 'crispdm' && (
        crispdm.data ? (
          <CrispDmWalkthrough
            report={crispdm.data}
            layout="stepper"
            slots={{
              data_understanding: eda.data ? (
                <Chart
                  kind="bar"
                  height={240}
                  data={eda.data.amount_bands}
                  x="band"
                  series={[{ key: 'fraud_rate_pct', label: 'Fraud rate (%)' }]}
                  yFormat={(v) => `${v.toFixed(2)}%`}
                  caption="Fraud rate by amount band."
                />
              ) : undefined,
              modeling: run ? (
                <ModelLeaderboard run={run} entityLabel="Detector" showParams />
              ) : undefined,
              evaluation: budget.data ? (
                <Chart
                  kind="line"
                  height={260}
                  data={budget.data}
                  x="alerts"
                  series={[
                    { key: 'precision_pct', label: 'Precision (%)' },
                    { key: 'recall_pct', label: 'Fraud caught (%)' },
                  ]}
                  xLabel="Alerts reviewed"
                  yFormat={(v) => `${v.toFixed(0)}%`}
                  caption="Precision and recall against review budget."
                />
              ) : undefined,
            }}
          />
        ) : (
          <Notice>{crispdm.error ?? 'Loading the CRISP-DM report…'}</Notice>
        )
      )}

      {tab === 'models' && (
        run ? (
          <>
            <Panel
              title="Detector tournament"
              description="Unsupervised detectors never see a label; supervised models do. Two baselines are included so the gain is legible."
            >
              <ModelLeaderboard
                run={run}
                entityLabel="Detector"
                columns={[
                  { key: 'avg_precision', label: 'Avg precision' },
                  { key: 'roc_auc', label: 'ROC-AUC' },
                  { key: 'recall_at_budget', label: 'Caught in 100',
                    format: (v) => `${(v * 100).toFixed(1)}%` },
                  { key: 'accuracy', label: 'Accuracy',
                    format: (v) => `${(v * 100).toFixed(3)}%` },
                ]}
                showParams
              />
            </Panel>

            <Panel
              title="Average precision vs accuracy"
              description="The same eight detectors, measured two ways."
            >
              <Chart
                kind="bar"
                horizontal
                height={320}
                data={run.models.map((m) => ({
                  model: m.name,
                  avg_precision: m.metrics.avg_precision,
                }))}
                x="model"
                series={[{ key: 'avg_precision', label: 'Average precision' }]}
                highlightKey={run.best_model ?? undefined}
                yFormat={(v) => v.toFixed(3)}
                caption="Average precision separates these detectors across three orders of magnitude. Accuracy puts every one of them between 99.78% and 99.95% — including the detector that catches nothing."
              />
            </Panel>
          </>
        ) : (
          <Notice>{metrics.error ?? 'Loading metrics…'}</Notice>
        )
      )}

      {tab === 'audit' && (
        <>
          <AuditPanel
            findings={audit.data ?? []}
            runtimeAssertions={(run?.audit?.runtime_assertions as never) ?? []}
            summary={{
              critical: run?.audit?.critical as number | undefined,
              warnings: run?.audit?.warnings as number | undefined,
              total: run?.audit?.total as number | undefined,
            }}
          />
          {provenance.data && <ProvenanceBadge provenance={provenance.data} />}
        </>
      )}
    </AppShell>
  );
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      padding: 'var(--space-4)', borderRadius: 'var(--radius)',
      borderLeft: '3px solid var(--status-warning)',
      background: 'var(--surface-1)', border: '1px solid var(--border)',
      color: 'var(--text-secondary)', fontSize: 'var(--text-sm)',
    }}>
      {children}
    </div>
  );
}
