# Primeros pasos — GeoCebada

Este tutorial está pensado para integrantes del equipo que trabajan principalmente con notebooks y ya tienen **Conda** instalado.

La idea es que todos usemos la misma línea de Python (**3.11**) y que el repositorio se instale como paquete editable. Así, los notebooks pueden importar `geocebada` directamente y cualquier cambio en `src/geocebada/` queda disponible sin copiar funciones ni modificar `sys.path`.

## 1. Clonar el repositorio

En una terminal:

```bash
git clone https://github.com/heri-espino/GeoCebada.git
cd GeoCebada
```

También se puede clonar con GitHub Desktop. Lo importante es terminar dentro de la carpeta raíz del repositorio, donde existen `pyproject.toml`, `environment.yml`, `src/`, `notebooks/` y `data/`.

## 2. Crear el ambiente Conda

Desde la raíz del repositorio:

```bash
conda env create -f environment.yml
```

Esto crea un ambiente llamado:

```text
geocebada
```

con Python 3.11.

Después actívalo:

```bash
conda activate geocebada
```

Verifica que estás usando la versión esperada:

```bash
python --version
```

Debe mostrar `Python 3.11.x`.

> No trabajes desde `base`. Activa `geocebada` antes de abrir Jupyter o ejecutar scripts del proyecto.

## 3. Instalar GeoCebada y sus dependencias

Con el ambiente `geocebada` activo y todavía desde la raíz del repositorio:

```bash
python -m pip install -e ".[dev,geo]"
```

La opción `-e` significa **editable**. El paquete no se copia a otro lugar: Python apunta al código del repositorio. Si alguien actualiza una función bajo `src/geocebada/`, los notebooks empiezan a usar esa versión sin reinstalar el proyecto.

`[dev,geo]` instala además las herramientas de trabajo del equipo, incluyendo JupyterLab, pytest y dependencias geoespaciales.

### Servidor con GPU NVIDIA — opcional

La mayor parte del proyecto funciona perfectamente sin GPU. Para la alineación temporal y futuros modelos con soporte CUDA existe un extra opcional.

Primero verifica que el servidor vea la GPU:

```bash
nvidia-smi
```

Si funciona y el servidor tiene un driver NVIDIA compatible con CUDA 12, instala:

```bash
python -m pip install -e ".[dev,geo,gpu]"
```

El extra `gpu` instala `cupy-cuda12x`. Para comprobarlo:

```bash
python -c "from geocebada.features import temporal_backend_available; print(temporal_backend_available('auto'))"
```

En una máquina CUDA correctamente configurada debería imprimir:

```text
gpu
```

Si imprime `cpu`, el resto del proyecto sigue funcionando normalmente. No instales el extra GPU en computadoras sin NVIDIA/CUDA sólo para mantener el entorno igual: el código está diseñado para que CPU y GPU produzcan la misma representación temporal.

## 4. Registrar el kernel de Jupyter

Ejecuta una vez:

```bash
python -m ipykernel install --user --name geocebada --display-name "Python (GeoCebada)"
```

Después, dentro de Jupyter, selecciona siempre el kernel:

```text
Python (GeoCebada)
```

Esto evita que el notebook se ejecute accidentalmente con `base` u otro ambiente.

## 5. Verificar que la instalación funciona

Ejecuta:

```bash
python -c "import geocebada; print(geocebada.__version__)"
```

Debería imprimir la versión del paquete, actualmente `0.1.0`.

También puedes verificar el repositorio con:

```bash
pytest -q
```

Si las pruebas pasan, el ambiente básico está correctamente configurado.

## 6. Abrir los notebooks

Con `geocebada` activo:

```bash
jupyter lab
```

Empieza por:

```text
notebooks/01_visualizacion_datos.ipynb
```

Después, para estudiar las series satelitales y construir una rejilla temporal común:

```text
notebooks/02_cobertura_alineacion_temporal.ipynb
```

Los notebooks contienen además una instalación editable defensiva para que puedan ejecutarse de forma sencilla. Si ya seguiste este tutorial, esa instalación es redundante pero inocua.

Dentro de Jupyter confirma que el kernel seleccionado sea **Python (GeoCebada)**.

## 7. Flujo normal de trabajo

Cada vez que vuelvas al proyecto:

```bash
cd GeoCebada
conda activate geocebada
git pull
jupyter lab
```

No es necesario recrear el ambiente todos los días.

## 8. Si cambian las dependencias

Si alguien modifica `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Si cambia `pyproject.toml` o se agregan nuevas dependencias del paquete:

```bash
python -m pip install -e ".[dev,geo]"
```

En el servidor GPU usa, en cambio:

```bash
python -m pip install -e ".[dev,geo,gpu]"
```

Después reinicia el kernel de Jupyter.

## 9. Comprobación rápida dentro de un notebook

Una celda útil al inicio de un notebook es:

```python
import sys

import geocebada

print("Python:", sys.version)
print("Entorno:", sys.executable)
print("GeoCebada:", geocebada.__version__)
```

La ruta de `sys.executable` debería apuntar al ambiente `geocebada` y la versión debería comenzar con `3.11`.

## 10. Problemas comunes

### `conda activate geocebada` no funciona

En Windows PowerShell, normalmente basta ejecutar una vez:

```powershell
conda init powershell
```

Cierra y vuelve a abrir PowerShell.

En macOS/Linux con zsh:

```bash
conda init zsh
```

Después reinicia la terminal.

### Jupyter no muestra `Python (GeoCebada)`

Activa el ambiente y vuelve a registrar el kernel:

```bash
conda activate geocebada
python -m ipykernel install --user --name geocebada --display-name "Python (GeoCebada)"
```

### `ModuleNotFoundError: No module named 'geocebada'`

Desde la raíz del repositorio y con el ambiente activo:

```bash
python -m pip install -e ".[dev,geo]"
```

Luego reinicia el kernel.

### El notebook usa otro Python

Comprueba:

```python
import sys
print(sys.executable)
```

Si no apunta al ambiente `geocebada`, cambia el kernel a **Python (GeoCebada)**.

### La GPU no aparece

Comprueba primero fuera de Python:

```bash
nvidia-smi
```

Después verifica CuPy:

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Si falla, usa `BACKEND="cpu"` mientras se revisa CUDA/driver. No bloquea el análisis.

## Resumen mínimo

Para una computadora nueva, el flujo completo es:

```bash
git clone https://github.com/heri-espino/GeoCebada.git
cd GeoCebada
conda env create -f environment.yml
conda activate geocebada
python -m pip install -e ".[dev,geo]"
python -m ipykernel install --user --name geocebada --display-name "Python (GeoCebada)"
jupyter lab
```

En el servidor NVIDIA puede sustituirse la instalación por:

```bash
python -m pip install -e ".[dev,geo,gpu]"
```

Después sólo hace falta activar el ambiente antes de trabajar.
