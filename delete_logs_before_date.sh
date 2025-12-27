ROOT="/PublicSSD/himchan/Memory-RL/logs"
cutoff="2025-12-01"

find "$ROOT" -type d -print0 |
while IFS= read -r -d '' d; do
  base="$(basename "$d")"
  if [[ "$base" =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then
    dt="${BASH_REMATCH[1]}"
    if [[ "$dt" < "$cutoff" ]]; then
      rm -rf -- "$d"
      echo "REMOVED: $d"
    fi
  fi
done