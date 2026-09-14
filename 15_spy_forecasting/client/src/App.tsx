/**
 * Project 15 client.
 *
 * ForecastFan is the only project-specific view. This project is the hardest
 * test of the shared components: the primary metric is lower-is-better, the
 * "models" include pure baselines that win, and the honest result is that
 * almost nothing is predictable. The components had to render that without
 * being designed for a happy answer.
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
  StatRow,
  StatTile,
  useApi,
  useAudit,
  useCrispDm,
  useMeta,
  useMetrics,
  useProvenance,
} from '@dsfs/ui';
import type { ModelScore, Tab } from '@dsfs/ui';
import { ForecastFan } from './ForecastFan';

const TABS: Tab[] = [
  { id: 'forecast', label: 'Forecast' },
  { id: 'backtest', label: 'Backtest' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'crispdm', label: 'CRISP-DM' },
  { id: 'models', label: 'Model tournament' },
  { id: 'audit', label: 'Audit' },
];

interface CurvePoint { date: string; strategy: number; buy_hold: number }
interface BacktestSummary {
  sharpe: number; ann_return_pct: number; max_drawdown_pct: number;
  time_in_market_pct: number; n_trades: number;
  buy_hold_sharpe: number; buy_hold_return_pct: number;
}
interface BacktestResponse {
  curves: Record<string, CurvePoint[]>;
  summary: Record<string, BacktestSummary>;
}
interface CalRow {
  model: string; coverage_80: number; interval_width_pct: number; pinball: number;
}
interface Significance {
  mean_diff: number; ci_low: number; ci_high: number; verdict: string;
}
interface CalResponse {
  calibration: CalRow[];
  significance_vs_random_walk: Record<string, Significance>;
  nominal_coverage: number;
}

export function App() {
  const [tab, setTab] = useState('forecast');
  const [selected, setSelected] = useState<string | null>(null);

  const meta = useMeta();
  const metrics = useMetrics();
  const crispdm = useCrispDm();
  const audit = useAudit();
  const provenance = useProvenance();
  const backtest = useApi<BacktestResponse>('/api/backtest');
  const cal = useApi<CalResponse>('/api/calibration');

  const run = metrics.data;
  const activeModel = selected ?? run?.best_model ?? null;
  const curve = activeModel ? backtest.data?.curves[activeModel] : undefined;
  const summary = activeModel ? backtest.data?.summary[activeModel] : undefined;

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
        <Notice>Run <code>python ml/train.py</code>, then reload.</Notice>
      )}

      {tab === 'forecast' && (
        <>
          <Panel
            title="Next-day forecast"
            description="A distribution, not a number. The width is the useful part."
          >
            <ForecastFan />
          </Panel>
          {run && (
            <Panel title="Walk-forward performance"
                   description="Across quarterly refits; every forecast is out-of-sample by construction.">
              <MetricsPanel
                run={run}
                highlight={[
                  { key: 'pinball', label: 'Pinball loss',
                    format: (v) => v.toFixed(6),
                    hint: 'Quantile loss across P10, P50 and P90. Lower is better.' },
                  { key: 'coverage_80', label: '80% coverage', unit: '%',
                    format: (v) => (v * 100).toFixed(1),
                    hint: 'Should be 80. This is the honest measure of an interval.' },
                  { key: 'hit_rate', label: 'Direction correct', unit: '%',
                    format: (v) => (v * 100).toFixed(1),
                    hint: '54.7% of days are up, so that is the bar.' },
                  { key: 'r2', label: 'R² on returns',
                    format: (v) => v.toFixed(4),
                    hint: 'Near zero is the expected, honest answer.' },
                ]}
              />
            </Panel>
          )}
        </>
      )}

      {tab === 'backtest' && (
        <>
          <Panel
            title="Strategy against buy-and-hold"
            description="Shown together deliberately: a strategy curve displayed alone is the oldest trick in quantitative marketing."
          >
            {run && (
              <div style={{ marginBottom: 'var(--space-4)' }}>
                <ModelLeaderboard
                  run={run}
                  entityLabel="Model"
                  selectable
                  selectedModel={activeModel}
                  onSelect={(m: ModelScore) => setSelected(m.name)}
                  columns={[
                    { key: 'pinball', label: 'Pinball', goodWhenUp: false },
                    { key: 'sharpe', label: 'Sharpe' },
                    { key: 'hit_rate', label: 'Hit rate',
                      format: (v) => `${(v * 100).toFixed(1)}%` },
                  ]}
                  maxRows={6}
                />
              </div>
            )}
            {curve ? (
              <Chart
                kind="line"
                height={320}
                data={curve}
                x="date"
                series={[
                  { key: 'strategy', label: `${activeModel} strategy` },
                  { key: 'buy_hold', label: 'Buy and hold SPY' },
                ]}
                xLabel="Date"
                yFormat={(v) => `${v.toFixed(2)}×`}
                caption="Growth of $1 over the walk-forward window, 2 bps round-trip cost. Select a model above to switch curves."
              />
            ) : (
              <Notice>{backtest.error ?? 'Loading equity curves…'}</Notice>
            )}
          </Panel>

          {summary && (
            <Panel title={`${activeModel}: traded result`}
                   description="Long whenever the median forecast is positive.">
              <StatRow>
                <StatTile label="Strategy Sharpe" value={summary.sharpe.toFixed(2)}
                          size="hero" accent="var(--series-1)" />
                <StatTile label="Buy-and-hold Sharpe"
                          value={summary.buy_hold_sharpe.toFixed(2)}
                          hint="The bar any strategy has to clear." />
                <StatTile label="Time in market" value={summary.time_in_market_pct.toFixed(0)}
                          unit="%" hint={summary.time_in_market_pct > 99
                            ? 'Always long — this is buy-and-hold wearing a model costume.'
                            : 'Fraction of days holding a position.'} />
                <StatTile label="Max drawdown" value={summary.max_drawdown_pct.toFixed(1)}
                          unit="%" deltaGoodWhenUp={false} />
                <StatTile label="Trades" value={summary.n_trades.toLocaleString()} />
              </StatRow>
            </Panel>
          )}
        </>
      )}

      {tab === 'calibration' && cal.data && (
        <>
          <Panel
            title="Are the intervals honest?"
            description="An 80% band that contains 80% of outcomes is useful. One that contains 55% is dangerous."
          >
            <Chart
              kind="bar"
              horizontal
              height={280}
              data={cal.data.calibration}
              x="model"
              series={[{ key: 'coverage_80', label: 'Actual coverage of the 80% band (%)' }]}
              yFormat={(v) => `${v.toFixed(1)}%`}
              caption="Nominal coverage is 80%. Below that the model is overconfident; above it, needlessly vague. The gradient-boosted model is the most overconfident here."
            />
          </Panel>

          <Panel
            title="Is any of this better than predicting zero?"
            description="Paired bootstrap on daily pinball loss, 5,000 resamples, against the random-walk baseline."
          >
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--text-sm)' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--axis)' }}>
                    <th style={th}>Model</th>
                    <th style={{ ...th, textAlign: 'right' }}>Loss difference</th>
                    <th style={{ ...th, textAlign: 'right' }}>95% CI</th>
                    <th style={th}>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(cal.data.significance_vs_random_walk).map(([name, s]) => {
                    const colour = s.verdict === 'better' ? 'var(--delta-up)'
                      : s.verdict === 'worse' ? 'var(--status-critical)'
                        : 'var(--text-muted)';
                    return (
                      <tr key={name} style={{ borderBottom: '1px solid var(--gridline)' }}>
                        <td style={td}>{name}</td>
                        <td style={{ ...td, textAlign: 'right' }}>{s.mean_diff.toExponential(2)}</td>
                        <td style={{ ...td, textAlign: 'right', color: 'var(--text-muted)' }}>
                          [{s.ci_low.toExponential(1)}, {s.ci_high.toExponential(1)}]
                        </td>
                        <td style={{ ...td, color: colour, fontWeight: 700 }}>
                          {s.verdict.toUpperCase()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p style={{
              marginTop: 'var(--space-3)', color: 'var(--text-secondary)',
              fontSize: 'var(--text-sm)', lineHeight: 1.5,
            }}>
              Negative means lower loss than a random walk. The gradient-boosted
              model — 26 features, three boosters, quarterly refits — lands on
              <strong> inconclusive</strong>: statistically indistinguishable
              from predicting exactly zero. The only model that reliably beats
              the random walk is the one that adds a constant drift, which is
              the equity risk premium rather than a forecast.
            </p>
          </Panel>
        </>
      )}

      {tab === 'crispdm' && (
        crispdm.data ? (
          <CrispDmWalkthrough
            report={crispdm.data}
            layout="stepper"
            slots={{
              modeling: run ? (
                <ModelLeaderboard run={run} entityLabel="Model" showParams />
              ) : undefined,
              evaluation: curve ? (
                <Chart
                  kind="line" height={260} data={curve} x="date"
                  series={[
                    { key: 'strategy', label: 'Strategy' },
                    { key: 'buy_hold', label: 'Buy and hold' },
                  ]}
                  xLabel="Date" yFormat={(v) => `${v.toFixed(2)}×`}
                  caption="Walk-forward equity against buy-and-hold."
                />
              ) : undefined,
            }}
          />
        ) : (
          <Notice>{crispdm.error ?? 'Loading…'}</Notice>
        )
      )}

      {tab === 'models' && (
        run ? (
          <Panel
            title="Model tournament"
            description="Baselines are not straw men here: a random walk is close to the best available answer for daily equity returns."
          >
            <ModelLeaderboard
              run={run}
              entityLabel="Model"
              selectable
              selectedModel={activeModel}
              onSelect={(m: ModelScore) => setSelected(m.name)}
              columns={[
                { key: 'pinball', label: 'Pinball loss', goodWhenUp: false },
                { key: 'coverage_80', label: '80% coverage',
                  format: (v) => `${(v * 100).toFixed(1)}%` },
                { key: 'hit_rate', label: 'Hit rate',
                  format: (v) => `${(v * 100).toFixed(1)}%` },
                { key: 'r2', label: 'R²', format: (v) => v.toFixed(4) },
                { key: 'sharpe', label: 'Sharpe' },
              ]}
              showParams
            />
          </Panel>
        ) : (
          <Notice>{metrics.error ?? 'Loading…'}</Notice>
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

const th: React.CSSProperties = {
  textAlign: 'left', padding: '8px 10px', color: 'var(--text-muted)',
  fontWeight: 600, fontSize: 'var(--text-xs)', textTransform: 'uppercase',
  letterSpacing: 0.3, whiteSpace: 'nowrap',
};
const td: React.CSSProperties = {
  padding: '9px 10px', color: 'var(--text-primary)', whiteSpace: 'nowrap',
};
