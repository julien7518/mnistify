import type { ComponentProps, HTMLAttributes } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
export type StatusProps = ComponentProps<typeof Badge> & {
    status: 'online' | 'offline';
};
export const Status = ({ className, status, ...props }: StatusProps) => (
    <Badge
        className={cn('flex items-center gap-2', 'group', status, className)}
        variant="secondary"
        {...props}
    />
);
export type StatusIndicatorProps = HTMLAttributes<HTMLSpanElement>;
export const StatusIndicator = ({
    ...props
}: StatusIndicatorProps) => (
    <span className="relative flex h-2 w-2" {...props}>
        <span
            className={cn(
                'absolute inline-flex h-full w-full animate-ping rounded-full opacity-75',
                'group-[.online]:bg-emerald-500',
                'group-[.offline]:bg-red-500'
            )}
        />
        <span
            className={cn(
                'relative inline-flex h-2 w-2 rounded-full',
                'group-[.online]:bg-emerald-500',
                'group-[.offline]:bg-red-500'
            )}
        />
    </span>
);
export type StatusLabelProps = HTMLAttributes<HTMLSpanElement> & {
    time?: number | null;
    error?: string | null;
};
export const StatusLabel = ({
    className,
    children,
    time,
    error,
    ...props
}: StatusLabelProps) => (
    <span className={cn('text-muted-foreground', className)} {...props}>
        {children ?? (
            <>
                <span className="hidden group-[.online]:block">Ready ({time} ms)</span>
                <span className="hidden group-[.offline]:block">Error {error && `(${error})`}</span>
            </>
        )}
    </span>
);