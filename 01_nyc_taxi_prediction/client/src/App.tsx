/**
 * Project 01 client.
 *
 * Almost everything here comes from @dsfs/ui. The only code specific to this
 * project is the zone map and the trip estimator; the rest -- metrics, the
 * tournament table, the CRISP-DM walkthrough, the audit -- is the shared,
 * API-driven machinery that all sixteen projects use.
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
import { TripEstimator } from './TripEstimator';
import { ZoneMap } from './ZoneMap';

const TABS: Tab[] = [
  { id: 'estimator', label: 'Trip estimator' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'models', label: 'Model tournament' },
  { id: 'audit', label: 'Audit' },
];

interface Importance {
  feature: string;
  gain: number;
  share: number;
}

export function App() {
  const [tab, setTab] = useState('estimator');

  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const importance = useApi<Importance[]>('/api/importance?top=12');

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
          No training artifacts yet. Run <code>python ml/train.py</code> in the
          project directory, then reload.
        </Notice>
      )}

      {tab === 'estimator' && (
        <>
          <Panel
            title="Estimate a trip"
            description="Uses only what is knowable when a rider books: origin, destination, time and party size."
          >
            <TripEstimator />
          </Panel>

          {run && (
            <Panel
              title="How the model performs"
              description="Measured on held-out days the model never saw during training."
            >
              <MetricsPanel
                run={run}
                highlight={[
                  { key: 'rmse', label: 'RMSE', unit: 'min',
                    format: (v) => v.toFixed(2),
                    hint: 'Typical error of a single estimate.' },
                  { key: 'within_5min', label: 'Within 5 min', unit: '%',
                    format: (v) => v.toFixed(1),
                    hint: 'The product-level question: how often it feels right.' },
                  { key: 'mae', label: 'MAE', unit: 'min',
                    format: (v) => v.toFixed(2) },
                  { key: 'r2', label: 'R²', format: (v) => v.toFixed(3) },
                ]}
              />
            </Panel>
          )}
        </>
      )}

      {tab === 'crispdm' && (
        crispdm.data ? (
          <CrispDmWalkthrough
            report={crispdm.data}
            layout="stepper"
            slots={{
              // The map belongs to this project; the walkthrough does not
              // know it exists.
              data_understanding: (
                <ZoneMap height={380} />
              ),
              modeling: run ? (
                <ModelLeaderboard run={run} entityLabel="Model" showParams />
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
              title="Model tournament"
              description="Every model saw identical features and an identical split, so the ranking reflects the estimator."
            >
              <ModelLeaderboard
                run={run}
                entityLabel="Model"
                columns={[
                  { key: 'rmse', label: 'RMSE', unit: 'min', goodWhenUp: false },
                  { key: 'mae', label: 'MAE', unit: 'min', goodWhenUp: false },
                  { key: 'r2', label: 'R²' },
                  { key: 'within_5min', label: 'Within 5 min', unit: '%' },
                ]}
                showParams
              />
            </Panel>

            <Panel
              title="Error by model"
              description="Baselines included, because a learned model has to beat the obvious solution to justify existing."
            >
              <Chart
                kind="bar"
                horizontal
                height={280}
                data={run.models.map((m) => ({
                  model: m.name,
                  rmse: m.metrics.rmse,
                }))}
                x="model"
                series={[{ key: 'rmse', label: 'RMSE (minutes)' }]}
                highlightKey={run.best_model ?? undefined}
                yFormat={(v) => `${v.toFixed(2)}`}
                caption="Lower is better. The winning model is highlighted; grey bars are the remaining entries."
              />
            </Panel>

            {importance.data && (
              <Panel
                title="What the model leans on"
                description="Gain-based importance from the fitted booster."
              >
                <Chart
                  kind="bar"
                  horizontal
                  height={340}
                  data={importance.data.map((f) => ({
                    feature: f.feature,
                    share: f.share,
                  }))}
                  x="feature"
                  series={[{ key: 'share', label: 'Share of total gain (%)' }]}
                  yFormat={(v) => `${v.toFixed(1)}%`}
                  caption="Route distance and the zone-pair encoding dominate — both reconstructed from the training window, since the meter reading is unavailable at booking."
                />
              </Panel>
            )}
          </>
        ) : (
          <Notice>{metrics.error ?? 'Loading metrics…'}</Notice>
        )
      )}

      {tab === 'audit' && (
        <>
          <AuditPanel
            findings={audit.data ?? []}
            runtimeAssertions={
              (run?.audit?.runtime_assertions as never) ?? []
            }
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
      padding: 'var(--space-4)',
      borderRadius: 'var(--radius)',
      borderLeft: '3px solid var(--status-warning)',
      background: 'var(--surface-1)',
      border: '1px solid var(--border)',
      color: 'var(--text-secondary)',
      fontSize: 'var(--text-sm)',
    }}>
      {children}
    </div>
  );
}
