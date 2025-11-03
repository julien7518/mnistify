"use client"

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"

import {
    ChartConfig,
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

const chartConfig = {
    proba: {
        label: "Probability",
        color: "#2563eb",
    }
} satisfies ChartConfig

interface predictionProps {
    number: number;
    proba: number;
}

export type chartDataProps = predictionProps[];

export function ChartView({ data }: { data: chartDataProps }) {
    return (
        <ChartContainer config={chartConfig} className="max-h-[150px] min-h-[150px] w-full">
            <BarChart accessibilityLayer data={data}>
                <CartesianGrid vertical={false} />
                <YAxis domain={[0, 100]} hide tickCount={3}/>
                <XAxis
                    dataKey="number"
                    tickLine={false}
                    tickMargin={10}
                    axisLine={false}
                />
                <ChartTooltip content={<ChartTooltipContent hideIndicator hideLabel />} />
                <Bar dataKey="proba" fill="var(--color-proba)" radius={4} />
            </BarChart>
        </ChartContainer>
    )
}
