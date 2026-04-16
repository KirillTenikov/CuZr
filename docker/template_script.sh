# Example local build commands for macOS laptop (Docker Desktop)

docker --context=default buildx build \
  --builder default \
  --platform linux/amd64 \
  --progress=plain \
  --provenance=false \
  --sbom=false \
  -f Dockerfile.cuzr_plan_c \
  -t local/cuzr-prelammps:cu126 \
  --load \
  .

# After launching an instance from this image:
# bash startup_md_lammps_only.sh
