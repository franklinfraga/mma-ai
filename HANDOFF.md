# MMA AI Fork Handoff

Data deste handoff: 2026-06-29.

Este arquivo resume o estado atual do fork `franklinfraga/mma-ai` e o caminho
mais curto para deixar treino e predicao funcionando com a GPU AMD Radeon RX
6800 XT via ROCm.

## TL;DR

O projeto esta praticamente funcional em CPU e com dados/modelo atualizados. O
bloqueio principal para GPU e o ambiente Python: a `.venv` atual esta com
`torch 2.9.1+cu128`, `torch.version.hip == None` e
`torch.cuda.is_available() == False`. Ou seja, ROCm enxerga a placa, mas o
PyTorch instalado e o wheel CUDA, puxado pelo `uv.lock`.

Prioridade imediata:

1. Fixar a fonte ROCm do PyTorch no `pyproject.toml`.
2. Regenerar `uv.lock`.
3. Reinstalar/sincronizar o ambiente.
4. Validar que `torch.version.hip` nao e `None` e que
   `torch.cuda.is_available()` retorna `True`.
5. Rodar predicao e, depois, retreinar.

## Estado Confirmado

Repositorio local:

- Caminho: `/home/frank/Projetos/mma-ai`
- GitHub: `https://github.com/franklinfraga/mma-ai`
- Dataset publico: `https://huggingface.co/datasets/franklinfraga/mma-ai`
- Backup antigo informado: `/run/media/frank/Backup/Projetos/mma-ai`

Mudancas pendentes no git:

- `CLAUDE.md`
- `README.md`
- `docker-compose.yml`
- `docs/HUGGINGFACE_DATASET.md`
- `main.py`
- `pyproject.toml`
- `setup.ps1`
- `setup.sh`
- `tests/test_setup_scripts.py`
- `uv.lock`

Arquivos gerados grandes presentes localmente:

- `data/prediction_data.csv` com aproximadamente 320 MB.
- `data/training_data.csv` com aproximadamente 283 MB.
- `data/training_data_dec.csv` com aproximadamente 411 MB.

Modelos presentes:

- `AutogluonModels/ag-20260304_110750-win-extreme`
- `AutogluonModels/ag-20260628_190935-win-extreme`

O Docker nao estava com servicos ativos no momento desta analise:

```bash
docker compose ps
```

retornou lista vazia.

## O Que Ja Foi Feito

Fork e setup:

- `setup.sh` aponta para
  `https://huggingface.co/datasets/franklinfraga/mma-ai/resolve/main`.
- `setup.ps1` aponta para o dataset do Franklin.
- `README.md`, `docs/HUGGINGFACE_DATASET.md` e `CLAUDE.md` foram ajustados para
  o fork/dataset novo.
- Hashes SHA256 fixos foram limpos dos artefatos do setup.
- `setup.sh` recebeu early return em `validate_manifest_artifact_pins()` quando
  nao ha pins configurados.
- `tests/test_setup_scripts.py` foi atualizado para a URL nova e para nao exigir
  o hash antigo do modelo.

Banco, dados e pipeline:

- Scraper UFCStats ja rodou com dados ate junho de 2026, segundo o historico
  informado.
- Bancos `mma-ai` e `odds` foram restaurados anteriormente, segundo o historico
  informado.
- Feature pipeline reconstruiu os CSVs finais em `data/`.

Modelo:

- Modelo novo informado: `ag-20260628_190935-win-extreme`.
- Accuracy informada: `65.66%`.
- Ensemble informado: RealTabPFN-v2, TabICL e LightGBM.

Correcoes de codigo/configuracao:

- `docker-compose.yml` agora tem `shm_size: '512m'` no servico `db`.
- `main.py` nao tem mais o import duplicado de `Path` dentro do bloco que
  causava `UnboundLocalError`.
- `pyproject.toml` ja recebeu `numba>=0.60.0`.

GPU/ROCm:

- `rocm-smi` detecta `AMD Radeon RX 6800 XT`.
- `rocminfo` detecta `gfx1030`.
- A placa esta visivel para ROCm, entao o problema restante esta no ambiente
  Python/PyTorch.

## Diagnostico Do Bloqueio ROCm

Validacao local da `.venv`:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.version, 'hip', None)); print(getattr(torch.version, 'cuda', None))"
```

Resultado observado:

```text
2.9.1+cu128
cuda_available False
hip None
cuda_version 12.8
```

Conclusao:

- O ambiente esta com PyTorch CUDA 12.8.
- Para ROCm, `torch.version.hip` precisa retornar uma versao HIP/ROCm.
- Em PyTorch com ROCm, a API continua sendo `torch.cuda`; portanto
  `torch.cuda.is_available()` tambem deve retornar `True`.
- O `uv.lock` atual contem pacotes `nvidia-*`, confirmando que o lock ainda esta
  resolvendo a arvore CUDA.

Checagem do indice oficial de wheels do PyTorch:

- `https://download.pytorch.org/whl/rocm6.3/torch/` contem
  `torch-2.9.1+rocm6.3`.
- `https://download.pytorch.org/whl/rocm6.4/torch/` tambem contem
  `torch-2.9.1+rocm6.4`.
- `https://download.pytorch.org/whl/rocm7.1/torch/` contem versoes mais novas
  observadas no indice, mas nao `2.9.1`.

Como o plano atual ja mira `torch==2.9.1+rocm6.3`, o caminho conservador e
comecar por `rocm6.3`. Se houver incompatibilidade pratica com ROCm 7.1 do
sistema, testar `rocm6.4` ou uma combinacao PyTorch/ROCm 7.x depois.

## Prioridade 1: Fixar PyTorch ROCm No `pyproject.toml`

Editar `pyproject.toml` para trocar a dependencia generica:

```toml
"torch>=2.5.1",
```

por uma dependencia ROCm pinada ou, no minimo, resolvida pelo indice ROCm. A
opcao mais previsivel para este ambiente e:

```toml
"torch==2.9.1+rocm6.3",
"pytorch-triton-rocm",
```

Adicionar ao final do arquivo:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.3"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-rocm" }
pytorch-triton-rocm = { index = "pytorch-rocm" }
```

Depois regenerar o lock:

```bash
rm uv.lock
uv lock
```

Se o resolver reclamar de `torch==2.9.1+rocm6.3`, tentar primeiro:

```toml
"torch>=2.9.1",
"pytorch-triton-rocm",
```

mantendo `[tool.uv.sources]` apontando `torch` para `pytorch-rocm`.

## Prioridade 2: Recriar/Sincronizar O Ambiente

Depois do lock ROCm:

```bash
uv sync
```

Se o ambiente mantiver restos CUDA, usar uma reinstalacao mais agressiva:

```bash
uv sync --reinstall
```

Validar:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda facade available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Resultado esperado:

- `torch` com sufixo `+rocm6.3` ou equivalente ROCm.
- `hip` preenchido.
- `cuda facade available: True`.
- Device parecido com `AMD Radeon RX 6800 XT`.

Se a GPU aparecer no ROCm mas PyTorch falhar em `gfx1030`, testar:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Se isso resolver, colocar a variavel no shell ou em um script local de treino.

## Workaround Ate O Lock Ficar Certo

Enquanto o `uv.lock` ainda estiver CUDA, evitar resync automatico:

```bash
uv run --no-sync mma-predict --odds --no-shap
uv run --no-sync mma-train
```

Esse workaround so preserva o que ja estiver instalado na `.venv`; ele nao
conserta a causa raiz.

## Prioridade 3: Predizer O Proximo Evento

Com PyTorch ROCm validado, rodar:

```bash
uv run mma-predict --odds --no-shap --no-manual-odds --prediction-model lightgbm
```

Se ainda estiver usando workaround:

```bash
uv run --no-sync mma-predict --odds --no-shap --no-manual-odds --prediction-model lightgbm
```

Saida esperada:

- `data/predictions/latest/fight_predictions.csv`

Status validado localmente:

- `uv run mma-predict --odds --no-shap --no-manual-odds --prediction-model lightgbm`
  concluiu com `fight_predictions.csv`.
- O modelo `ag-20260628_190935-win-extreme` vem sem `feats.txt`; para usa-lo,
  eh preciso reaproveitar a lista de features do modelo anterior ou regenerar o
  arquivo no artefato.

Se BestFightOdds bloquear ou demorar:

- Rodar sem odds live para testar o pipeline basico.
- Usar odds manuais/API pelo dashboard quando necessario.
- O toggle Flaresolverr e apenas para bloqueio do BestFightOdds.

## Prioridade 4: Retreinar Com GPU

Quando `torch.cuda.is_available()` retornar `True`:

```bash
uv run mma-train
```

O treino deve usar GPU automaticamente nos modelos compativeis. Se ainda houver
risco de sync indevido:

```bash
uv run --no-sync mma-train
```

Depois avaliar:

```bash
uv run mma-evaluate --write-report --format text
```

Verificar no log do treino se modelos neural/TabPFN/TabICL/MITRA estao usando
GPU e se MITRA deixa de pular por memoria.

## Prioridade 5: Testes Antes Do Commit

Para as mudancas atuais de setup/docs:

```bash
uv run pytest tests/test_setup_scripts.py -q
uv run pytest tests/test_web/test_release_docs.py -q
```

Para uma checagem maior:

```bash
uv run mma-release-audit
```

Se mudar comportamento do dashboard ou predicao:

```bash
uv run pytest tests/test_web -q
uv run pytest tests/test_inference -q
```

## Prioridade 6: Commit E Push

Antes de commitar:

```bash
git status --short
git diff --stat
```

Nao adicionar artefatos gerados:

- `data/*.csv` grandes gerados
- `data/predictions/`
- `AutogluonModels/`
- dumps de banco
- logs
- `.env`
- screenshots

Commit sugerido depois do fix ROCm e testes:

```bash
git add setup.sh setup.ps1 README.md docs/HUGGINGFACE_DATASET.md CLAUDE.md \
  tests/test_setup_scripts.py docker-compose.yml main.py pyproject.toml uv.lock HANDOFF.md
git commit -m "Configurar fork e ambiente ROCm"
git push origin main
```

## Prioridade 7: Atualizar Hugging Face Dataset

Fazer isso so depois de validar o modelo novo:

1. Gerar dumps dos bancos.
2. Copiar CSVs finais de `data/`.
3. Compactar o novo modelo em `AutogluonModels/`.
4. Atualizar o dataset em
   `https://huggingface.co/datasets/franklinfraga/mma-ai`.
5. Preencher novamente os SHA256 nos scripts se quiser voltar a verificar pins
   fortes no setup publico.

Exemplo de dump:

```bash
mkdir -p dumps
docker compose exec db pg_dump --format=custom -U postgres mma-ai > dumps/mma-ai.postgres-custom
docker compose exec db pg_dump --format=custom -U postgres odds > dumps/odds.postgres-custom
```

## Comandos Uteis

Subir banco e dashboard:

```bash
docker compose up -d db web
```

Rodar dashboard local sem Docker:

```bash
uv run mma-web
```

Scrape incremental UFCStats:

```bash
uv run mma-scrape-ufcstats
```

Reconstruir features sem scrape live de odds:

```bash
uv run mma-rebuild-db --scrape --reset-db --odds-features
```

Predizer:

```bash
uv run mma-predict --odds --no-shap
```

Treinar:

```bash
uv run mma-train
```

## Regras Importantes Do Repositorio

- Usar `uv` como runner Python.
- Nao adicionar controles de treino no dashboard; treino e fluxo CLI.
- Acoes longas do dashboard devem continuar como background jobs.
- Analytics devem permanecer read-only.
- Nao adicionar CDN publico para Plotly ou icones.
- Manter Postgres em `postgres:18.1`.
- Manter `docker/postgres-init/01-create-odds.sql` montado para criar o banco
  `odds`.
- Se mudar setup publico, Docker, dashboard assets ou restore de artefatos,
  atualizar docs e testes relacionados.

## Proximo Passo Recomendado

Fazer primeiro somente o fix do `pyproject.toml`/`uv.lock` para ROCm e validar
PyTorch. Quando `torch.version.hip` aparecer e `torch.cuda.is_available()` der
`True`, o restante do projeto ja tem dados, modelo, scripts e comandos para
seguir para predicao e retreino.
