export default function FollowUpBar({ canFollowUp, armed, onArm }) {
  if (!canFollowUp) return null;
  return (
    <div style={{ marginTop: 12, marginBottom: 4 }}>
      {!armed ? (
        <button
          onClick={onArm}
          title="Następne pytanie potraktuję jako dopytanie (dołożę TOP-1 ustęp)."
          className="followup-btn"
        >
          Chciałbym dopytać
        </button>
      ) : (
        <span style={{ marginLeft: 8, fontSize: 12 }}>
          Tryb dopytania włączony — kolejne pytanie będzie follow-upem.
        </span>
      )}
    </div>
  );
}
