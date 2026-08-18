export function ComingSoonView({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] text-center gap-2">
      <h2 className="text-[16px] font-semibold text-on-surface">{title}</h2>
      <p className="text-[13px] text-secondary max-w-sm">
        This view lands in Part 2, once the RAG pipeline and agent are built. The system currently
        observes and predicts; it does not yet act autonomously.
      </p>
    </div>
  );
}
