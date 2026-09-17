/**
 * @dsfs/ui -- shared design system and data-science UI primitives.
 *
 * Import the tokens once in a project's entry point:
 *   import '@dsfs/ui/styles.css';
 *
 * Six components are driven entirely by API data and shared across the
 * portfolio: MetricsPanel, ModelLeaderboard, CrispDmWalkthrough, AuditPanel,
 * PredictionForm and Chart. None of them computes an aggregate of its own --
 * every figure they render comes from metrics.json or an API response, so any
 * published number traces back to an artifact.
 */

export * from './types';
export * from './hooks/useApi';

/* -- frame ---------------------------------------------------------- */
export { AppShell, Panel, useTheme } from './components/AppShell';
export type { AppShellProps, Tab, Theme } from './components/AppShell';

/* -- the six shared, data-driven components -------------------------- */
export { MetricsPanel } from './components/MetricsPanel';
export type { MetricsPanelProps, MetricSpec } from './components/MetricsPanel';

export { ModelLeaderboard, modelColors } from './components/ModelLeaderboard';
export type { ModelLeaderboardProps, ColumnSpec } from './components/ModelLeaderboard';

export { CrispDmWalkthrough } from './components/CrispDmWalkthrough';
export type { CrispDmWalkthroughProps } from './components/CrispDmWalkthrough';

export { AuditPanel } from './components/AuditPanel';
export type { AuditPanelProps, ExecutedAssertion } from './components/AuditPanel';

export { PredictionForm } from './components/PredictionForm';
export type { PredictionFormProps, FieldSpec, Option } from './components/PredictionForm';

export { Chart, seriesColor } from './components/Chart';
export type { ChartProps, ChartSeries } from './components/Chart';

/* -- supporting primitives ------------------------------------------- */
export { ProvenanceBadge } from './components/ProvenanceBadge';
export type { ProvenanceBadgeProps } from './components/ProvenanceBadge';

export { StatTile, StatRow } from './components/StatTile';
export type { StatTileProps } from './components/StatTile';

export { MarkdownMath } from './components/MarkdownMath';
