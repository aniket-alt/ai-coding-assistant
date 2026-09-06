/**
 * @dsfs/ui -- shared design system and data-science UI primitives.
 *
 * Import the tokens once in a project's entry point:
 *   import '@dsfs/ui/styles.css';
 */

export * from './types';
export * from './hooks/useApi';

export { AppShell, Panel, useTheme } from './components/AppShell';
export type { AppShellProps, Tab, Theme } from './components/AppShell';

export { ProvenanceBadge } from './components/ProvenanceBadge';
export type { ProvenanceBadgeProps } from './components/ProvenanceBadge';

export { StatTile, StatRow } from './components/StatTile';
export type { StatTileProps } from './components/StatTile';

export { Leaderboard, seriesColor } from './components/Leaderboard';
export type { LeaderboardProps } from './components/Leaderboard';

export { AuditPanel } from './components/AuditPanel';
export { CrispDmPanel } from './components/CrispDmPanel';
export { MarkdownMath } from './components/MarkdownMath';
