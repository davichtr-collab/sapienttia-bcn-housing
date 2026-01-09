# 🧠 SAPIENTTIA - Housing Policy Analysis Platform

Plataforma de análisis de políticas de vivienda para Barcelona con inteligencia artificial integrada.

## 🎯 Características

- **Simulación en tiempo real** de políticas de vivienda
- **Análisis con IA** usando Google Gemini
- **Datos reales** de Open Data Barcelona
- **Visualizaciones interactivas** con Plotly
- **Modelo económico** basado en elasticidades de mercado

## 🚀 Instalación

### Opción 1: Instalación Local
```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/sapienttia.git
cd sapienttia

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run sapienttia_app.py
```

### Opción 2: Despliegue en Streamlit Cloud

1. Haz fork de este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Añade tu API Key de Google Gemini en los Secrets

## 🔑 Configuración de API Key

Para usar el análisis con IA, necesitas una API Key de Google Gemini:

1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Crea una API Key gratuita
3. Configúrala de una de estas formas:

**Opción A: Archivo `.streamlit/secrets.toml` (local)**
```toml
GEMINI_API_KEY = "tu_api_key_aquí"
```

**Opción B: Variable de entorno**
```bash
export GEMINI_API_KEY="tu_api_key_aquí"
```

**Opción C: Input en la aplicación**
- Usa el campo de texto en la barra lateral

## 📊 Fuentes de Datos

- [Open Data BCN](https://opendata-ajuntament.barcelona.cat/) - Precios de alquiler
- Ajuntament de Barcelona - Censo de vivienda turística
- INE - Ingresos medios por hogar
- MIT Urban Economics Lab - Estudios de elasticidad

## 🛠️ Tecnologías Utilizadas

- **Streamlit** - Framework de aplicación web
- **Pandas & NumPy** - Procesamiento de datos
- **Plotly** - Visualizaciones interactivas
- **Google Gemini AI** - Análisis con inteligencia artificial

## 📖 Uso

1. **Ajusta los sliders** en la barra lateral:
   - Pisos turísticos a eliminar
   - Tope de reducción de alquiler
   - Inversión pública

2. **Observa el impacto** en tiempo real:
   - Precio medio de alquiler
   - Viviendas disponibles
   - Accesibilidad estimada

3. **Genera análisis con IA** para obtener insights profesionales

## 📄 Licencia

MIT License - Ver archivo [LICENSE](LICENSE)

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@tu_usuario](https://github.com/tu_usuario)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📮 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

⭐ Si te gusta este proyecto, ¡dale una estrella en GitHub!# sapienttia-bcn-housing
