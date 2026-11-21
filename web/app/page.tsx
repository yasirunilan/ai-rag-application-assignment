import QueryBox from "@/components/QueryBox";
import AnswerPanel from "@/components/AnswerPanel";
import SourcesPanel from "@/components/SourcesPanel";
import StatusBadge from "@/components/StatusBadge";

export default function Home() {
  return (
    <main style={{ padding: 24 }}>
      <StatusBadge />
      <QueryBox />
      <AnswerPanel />
      <SourcesPanel />
    </main>
  );
}
