import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import os
import google.generativeai as genai
import tempfile # Para manejar archivos subidos temporalmente
import pandas as pd

# --- Langchain & Google Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader # Añadido Unstructured como opción
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuración de la Base de Datos ---
DB_FILE = "app_data.db"
FAISS_INDEX_PATH = "faiss_index" # Directorio para guardar el índice FAISS

# --- Configuración de Gemini (Usando Streamlit Secrets) ---
GOOGLE_API_KEY = None
try:
    # Intentar obtener la clave desde los secretos de Streamlit
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google API Key cargada desde st.secrets.")
except KeyError:
    st.error("⚠️ **Error Crítico:** GOOGLE_API_KEY no encontrada en los secretos de Streamlit (`.streamlit/secrets.toml`). La funcionalidad del chat no estará disponible.")
    # Podrías añadir una forma de ingresarla manualmente si es para pruebas locales:
    # GOOGLE_API_KEY = st.text_input("Ingresa tu Google API Key:", type="password")
    # if GOOGLE_API_KEY:
    #     genai.configure(api_key=GOOGLE_API_KEY)
    # else:
    #     st.stop() # Detener la ejecución si no hay clave
except Exception as e:
     st.error(f"⚠️ Error configurando Google API: {e}")
     # st.stop()

# --- Funciones de Base de Datos (sin cambios respecto al código anterior) ---
def init_db():
    """Inicializa la base de datos y crea las tablas si no existen."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            dni TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dni TEXT,
            timestamp DATETIME,
            query TEXT,
            response TEXT,
            FOREIGN KEY (dni) REFERENCES usuarios (dni)
        )
    ''')
    cursor.execute("SELECT dni FROM usuarios WHERE dni = ?", ('1',))
    if not cursor.fetchone():
        hashed_password = hash_password('1')
        try:
            cursor.execute("INSERT INTO usuarios (dni, password_hash) VALUES (?, ?)", ('1', hashed_password))
            print("Usuario de prueba (1/1) creado.")
        except sqlite3.IntegrityError:
            print("El usuario de prueba ya existe.")
    conn.commit()
    conn.close()
    print(f"Base de datos '{DB_FILE}' inicializada.")

def hash_password(password):
    """Genera un hash SHA-256 para la contraseña."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verifica si la contraseña proporcionada coincide con el hash almacenado."""
    return stored_hash == hash_password(provided_password)

def get_user_hash(dni):
    """Obtiene el hash de la contraseña para un DNI específico."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM usuarios WHERE dni = ?", (dni,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def update_password(dni, new_password):
    """Actualiza la contraseña (hash) para un DNI específico."""
    new_hash = hash_password(new_password)
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE usuarios SET password_hash = ? WHERE dni = ?", (new_hash, dni))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error al actualizar contraseña para DNI {dni}: {e}")
        return False

def add_log(dni, query, response):
    """Añade una entrada al log de interacciones."""
    timestamp = datetime.now()
    # Limitar longitud de query/response si es necesario para la DB
    max_len = 5000 # Ajustar según necesidad
    query = query[:max_len]
    response = response[:max_len]
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO logs (dni, timestamp, query, response) VALUES (?, ?, ?, ?)",
                       (dni, timestamp, query, response))
        conn.commit()
        conn.close()
    except Exception as e:
         print(f"Error al añadir log para DNI {dni}: {e}")

# --- Funciones RAG (Retrieval-Augmented Generation) ---

# Usar cache para evitar recargar modelos/splitters innecesariamente
@st.cache_resource
def get_embeddings_model():
    """Carga el modelo de embeddings de Google."""
    print("Cargando modelo de embeddings...")
    if not GOOGLE_API_KEY:
         st.error("No se puede cargar el modelo de embeddings sin la API Key.")
         return None
    try:
        # Usar text-embedding-004 si está disponible y funciona, sino probar con embedding-001
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY) # o "models/text-embedding-004"
    except Exception as e:
        st.error(f"Error al cargar el modelo de embeddings: {e}")
        return None

@st.cache_resource
def get_llm():
    """Carga el modelo de lenguaje Gemini."""
    print("Cargando modelo LLM (Gemini)...")
    if not GOOGLE_API_KEY:
         st.error("No se puede cargar el LLM sin la API Key.")
         return None
    try:
        # Ajusta temperature y top_p según necesites
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY,
                                      temperature=0.2, top_p=0.5)
    except Exception as e:
        st.error(f"Error al cargar el LLM Gemini: {e}")
        return None


def fetch_logs_as_dataframe():
    """Lee la tabla de logs y la devuelve como un DataFrame de Pandas."""
    try:
        conn = sqlite3.connect(DB_FILE)
        # Usar parse_dates para intentar convertir la columna timestamp
        df = pd.read_sql_query("SELECT dni, timestamp, query, response FROM logs ORDER BY timestamp DESC",
                               conn,
                               parse_dates=['timestamp'])
        conn.close()

        # --- Manejo de Zona Horaria (IMPORTANTE) ---
        # SQLite no maneja zonas horarias nativamente. Asumiremos que se guardó en UTC
        # o como texto sin zona horaria. Vamos a convertirlo a la hora de Buenos Aires.
        if not df.empty and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dt.tz is None:
                print("Log timestamp es naive. Asumiendo UTC y convirtiendo a Buenos_Aires.")
                # Localiza a UTC y luego convierte a la zona deseada
                try:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert('America/Argentina/Buenos_Aires')
                except Exception as tz_err:
                     print(f"Error en conversión de zona horaria: {tz_err}. Se mostrará la hora original.")
                     # Podrías querer manejar esto de otra forma si la localización falla
            else:
                print("Log timestamp ya tiene zona horaria. Convirtiendo a Buenos_Aires.")
                # Si ya tiene zona horaria (ej. UTC), simplemente convierte
                try:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('America/Argentina/Buenos_Aires')
                except Exception as tz_err:
                     print(f"Error en conversión de zona horaria: {tz_err}. Se mostrará la hora original.")

        else:
             # Si la columna no es datetime (error en parse_dates o tabla vacía), intentar conversión manual
             try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Repetir lógica de zona horaria si la conversión manual funciona
                if df['timestamp'].dt.tz is None:
                     df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert('America/Argentina/Buenos_Aires')
                else:
                     df['timestamp'] = df['timestamp'].dt.tz_convert('America/Argentina/Buenos_Aires')
             except Exception as format_err:
                  print(f"No se pudo convertir 'timestamp' a datetime: {format_err}")
                  # Dejar la columna como está si falla la conversión

        return df
    except sqlite3.Error as e_sql:
         st.error(f"Error de Base de Datos al leer los logs: {e_sql}")
         return pd.DataFrame() # Devuelve DataFrame vacío en caso de error
    except Exception as e:
        st.error(f"Error general al leer o procesar los logs: {e}")
        import traceback
        print("--- TRACEBACK ERROR LEYENDO LOGS ---")
        traceback.print_exc()
        print("------------------------------------")
        return pd.DataFrame()


def reports_page():
    """Muestra la página de informes con estadísticas de uso."""
    st.header("Informes de Uso del Chatbot")

    df_logs = fetch_logs_as_dataframe()

    if df_logs.empty:
        st.warning("Aún no hay datos de logs para mostrar.")
        return

    st.success(f"Total de interacciones registradas: {len(df_logs)}")

    # --- Preparación de Datos ---
    # Necesitamos columnas separadas para fecha y hora para agrupar
    try:
        # Asegurarse de que timestamp sea datetime antes de extraer
        if not pd.api.types.is_datetime64_any_dtype(df_logs['timestamp']):
             df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'], errors='coerce') # Convertir, poner NaT si falla

        # Eliminar filas donde la conversión falló
        df_logs.dropna(subset=['timestamp'], inplace=True)

        if df_logs.empty:
             st.error("No se pudieron procesar las fechas de los logs.")
             return

        df_logs['fecha'] = df_logs['timestamp'].dt.date
        df_logs['hora'] = df_logs['timestamp'].dt.hour
        df_logs['dia_semana'] = df_logs['timestamp'].dt.day_name() # Opcional: día de la semana
    except Exception as e:
        st.error(f"Error al extraer fecha/hora de los logs: {e}")
        st.subheader("Detalle Completo (Error procesando fechas)")
        st.dataframe(df_logs)
        return

    # --- Agregaciones y Visualizaciones ---

    st.subheader("Interacciones por Día")
    daily_counts = df_logs.groupby('fecha').size().reset_index(name='Consultas')
    st.dataframe(daily_counts.sort_values(by='fecha', ascending=False), use_container_width=True)
    if not daily_counts.empty:
        try:
            # Convertir fecha a string para el gráfico para evitar problemas de tipo
            daily_counts_chart = daily_counts.copy()
            daily_counts_chart['fecha'] = pd.to_datetime(daily_counts_chart['fecha']).dt.strftime('%Y-%m-%d')
            st.bar_chart(daily_counts_chart.set_index('fecha'), y='Consultas')
        except Exception as chart_err:
             print(f"Error generando gráfico diario: {chart_err}")


    st.subheader("Interacciones por Hora del Día")
    hourly_counts = df_logs.groupby('hora').size().reset_index(name='Consultas')
    # Asegurar que todas las horas 0-23 estén presentes para un gráfico completo
    all_hours = pd.DataFrame({'hora': range(24)})
    hourly_counts = pd.merge(all_hours, hourly_counts, on='hora', how='left').fillna(0)
    hourly_counts['Consultas'] = hourly_counts['Consultas'].astype(int) # Convertir a entero
    st.dataframe(hourly_counts, use_container_width=True)
    if not hourly_counts.empty:
        st.bar_chart(hourly_counts.set_index('hora'))


    st.subheader("Interacciones por DNI (Usuario)")
    dni_counts = df_logs.groupby('dni').size().reset_index(name='Consultas')
    st.dataframe(dni_counts.sort_values(by='Consultas', ascending=False), use_container_width=True)


    # --- Detalle Completo ---
    st.subheader("Detalle Completo de Logs")
    with st.expander("Mostrar / Ocultar Todas las Interacciones"):
        # Mostrar columnas relevantes y formatear timestamp
        df_display = df_logs.copy()
        # Formatear la columna de timestamp para mejor legibilidad
        df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        st.dataframe(df_display[['timestamp', 'dni', 'query', 'response', 'fecha', 'hora']].sort_values(by='timestamp', ascending=False), use_container_width=True)
        
def load_documents2(uploaded_files):
    """Carga documentos desde archivos subidos (PDF, DOCX) - Prioriza Unstructured."""
    documents = []
    print(f"Cargando {len(uploaded_files)} archivos...")
    unstructured_available = False
    try:
        # Intentar importar Unstructured para ver si está disponible
        from langchain_community.document_loaders import UnstructuredFileLoader # Usar import moderno si es posible
        unstructured_available = True
        print("UnstructuredFileLoader está disponible.")
    except ImportError:
        print("UnstructuredFileLoader no encontrado, se usará Docx2txtLoader para DOCX si es necesario.")


    for uploaded_file in uploaded_files:
        # ... (creación de tempfile igual que antes) ...
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            print(f"Procesando archivo: {uploaded_file.name} (Tipo: {uploaded_file.type})")
            loader = None

            if uploaded_file.type == "application/pdf":
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    print(f"Usando PyPDFLoader para {uploaded_file.name}")
                except Exception as e_pdf:
                    st.error(f"Error con PyPDFLoader para {uploaded_file.name}: {e_pdf}")

            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                if unstructured_available:
                    try:
                        # Usar Unstructured si está disponible
                        loader = UnstructuredFileLoader(tmp_file_path)
                        print(f"Intentando con UnstructuredFileLoader para {uploaded_file.name}")
                    except Exception as e_unstructured:
                        st.error(f"Error con UnstructuredFileLoader para {uploaded_file.name}: {e_unstructured}")
                        # Podríamos intentar Docx2txt como fallback aquí si quisiéramos
                else:
                    # Usar Docx2txt si Unstructured no está
                    try:
                        loader = Docx2txtLoader(tmp_file_path)
                        print(f"Intentando con Docx2txtLoader para {uploaded_file.name}")
                    except ImportError:
                         st.error("Docx2txtLoader no está disponible.")
                    except Exception as e_docx2txt:
                         st.error(f"Error con Docx2txtLoader para {uploaded_file.name}: {e_docx2txt}")

            else:
                st.warning(f"Archivo no soportado: {uploaded_file.name} ({uploaded_file.type})")

            if loader:
                 try:
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"Documento '{uploaded_file.name}' cargado ({len(loaded_docs)} partes).")
                 except Exception as e_load:
                     st.error(f"Error al cargar el contenido de '{uploaded_file.name}': {e_load}")
                     if "zip file" in str(e_load).lower():
                          st.warning(f"El error 'not a zip file' sugiere que '{uploaded_file.name}' podría estar corrupto o no ser un DOCX válido. Intenta abrirlo y guardarlo de nuevo en Word.")

            # Eliminar archivo temporal
            os.remove(tmp_file_path)

        except Exception as e:
            st.error(f"Error general procesando el archivo {uploaded_file.name}: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                 os.remove(tmp_file_path)

    print(f"Total de documentos cargados: {len(documents)}")
    return documents

def load_documents(uploaded_files):
    """Carga documentos desde archivos subidos (PDF, DOCX)."""
    documents = []
    print(f"Cargando {len(uploaded_files)} archivos...")
    for uploaded_file in uploaded_files:
        try:
            # Guardar temporalmente para que los loaders puedan leerlo por path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            print(f"Procesando archivo: {uploaded_file.name}")
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                 # Docx2txtLoader suele ser simple, Unstructured es más potente pero pesado
                 # Probar con Docx2txtLoader primero
                 try:
                     loader = Docx2txtLoader(tmp_file_path)
                 except ImportError:
                      st.warning("Docx2txtLoader no disponible, intentando con UnstructuredFileLoader (puede requerir dependencias adicionales).")
                      # Instalar: pip install unstructured libmagic python-magic-bin
                      try:
                          loader = UnstructuredFileLoader(tmp_file_path)
                      except ImportError:
                           st.error("Por favor instala 'unstructured', 'libmagic' y 'python-magic-bin' para procesar DOCX con UnstructuredFileLoader.")
                           loader = None
                      except Exception as e_unstructured:
                           st.error(f"Error con UnstructuredFileLoader para {uploaded_file.name}: {e_unstructured}")
                           loader = None

            else:
                st.warning(f"Archivo no soportado: {uploaded_file.name} ({uploaded_file.type})")
                loader = None

            if loader:
                 try:
                    documents.extend(loader.load())
                    print(f"Documento '{uploaded_file.name}' cargado.")
                 except Exception as e_load:
                     st.error(f"Error al cargar el contenido de '{uploaded_file.name}': {e_load}")

            # Eliminar archivo temporal
            os.remove(tmp_file_path)

        except Exception as e:
            st.error(f"Error procesando el archivo {uploaded_file.name}: {e}")
            # Asegurar borrado si falla antes
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                 os.remove(tmp_file_path)

    print(f"Total de documentos cargados: {len(documents)}")
    return documents

def get_text_chunks(documents):
    """Divide los documentos en fragmentos más pequeños."""
    print("Dividiendo documentos en fragmentos...")
    # Ajustar chunk_size y chunk_overlap según sea necesario
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # Solapamiento para mantener contexto entre chunks
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Número total de fragmentos: {len(chunks)}")
    return chunks

def create_or_update_vectorstore(text_chunks, embeddings_model):
    """Crea o actualiza el vector store FAISS."""
    if not text_chunks:
        st.warning("No hay fragmentos de texto para procesar.")
        return None

    if not embeddings_model:
        st.error("Modelo de Embeddings no disponible. No se puede crear el Vector Store.")
        return None

    print("Creando/Actualizando Vector Store FAISS...")
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            # Cargar índice existente y añadir nuevos chunks
            print("Índice FAISS existente encontrado. Añadiendo nuevos documentos...")
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
            vectorstore.add_documents(text_chunks)
            print("Nuevos documentos añadidos al índice.")
        else:
            # Crear nuevo índice desde cero
            print("Creando nuevo índice FAISS...")
            vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings_model)
            print("Nuevo índice creado.")

        # Guardar el índice actualizado (o nuevo)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Índice FAISS guardado en '{FAISS_INDEX_PATH}'.")
        return vectorstore
    except Exception as e:
        st.error(f"Error al crear/actualizar el Vector Store FAISS: {e}")
        return None

# Usar cache para evitar recargar el índice repetidamente si no cambia
# @st.cache_resource # Ojo: Cachear esto puede ser complejo si se actualiza en admin_page
def load_vectorstore():
    """Carga el índice FAISS desde el disco."""
    embeddings_model = get_embeddings_model()
    if not embeddings_model:
        st.error("Modelo de Embeddings no disponible. No se puede cargar el Vector Store.")
        return None

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Cargando índice FAISS desde '{FAISS_INDEX_PATH}'...")
        try:
            # Necesario para cargar índices FAISS guardados localmente por Langchain
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
            print("Índice FAISS cargado correctamente.")
            return vectorstore
        except Exception as e:
            st.error(f"Error al cargar el índice FAISS: {e}. Puede que necesite ser regenerado.")
            return None
    else:
        print("Índice FAISS no encontrado.")
        return None

# Usar cache para el chain, se invalida si el vectorstore cambia (idealmente)
@st.cache_resource(show_spinner="Configurando asistente de IA...")
def get_qa_chain(_vectorstore): # Pasar vectorstore como argumento para cache
    """Crea la cadena de QA con el LLM y el retriever."""
    llm = get_llm()
    if not llm:
        st.error("LLM (Gemini) no disponible. La cadena de QA no puede ser creada.")
        return None
    if not _vectorstore:
         st.error("Vector Store no disponible. La cadena de QA no puede ser creada.")
         return None

    print("Creando cadena de QA...")
    # Crear un retriever desde el vector store
    # search_kwargs={'k': 4} -> obtener los 4 chunks más relevantes
    retriever = _vectorstore.as_retriever(search_kwargs={'k': 4})

    # Definir el prompt template
    # Ajusta este prompt según tus necesidades para guiar mejor a Gemini
    prompt_template = """Eres un asistente de Recursos Humanos muy útil. Tu tarea es responder preguntas sobre normativas, procedimientos y beneficios de la empresa basándote **únicamente** en el siguiente contexto proporcionado. Sé claro y conciso. Si la respuesta no se encuentra en el contexto, indica explícitamente "No tengo información sobre eso en los documentos proporcionados". No inventes respuestas.

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta útil:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Crear la cadena RetrievalQA
    # chain_type="stuff" -> Pone todos los chunks recuperados en el prompt (puede fallar si son demasiados)
    # chain_type="map_reduce", "refine" -> Opciones para contextos más largos
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Probar este primero
        retriever=retriever,
        return_source_documents=False, # Opcional: para ver qué chunks usó
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    print("Cadena de QA creada.")
    return qa_chain


# --- Lógica de la Aplicación Streamlit ---

# Inicializar estado de sesión (sin cambios)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'dni' not in st.session_state:
    st.session_state.dni = None
if 'page' not in st.session_state:
    st.session_state.page = 'Chat'

# --- Funciones de Página (UI) ---

def login_page():
    # (Sin cambios respecto al código anterior)
    st.header("Iniciar Sesión")
    dni_input = st.text_input("DNI")
    password_input = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        if not dni_input or not password_input:
            st.error("Por favor, ingrese DNI y Contraseña.")
            return
        stored_hash = get_user_hash(dni_input)
        if stored_hash and verify_password(stored_hash, password_input):
            st.session_state.logged_in = True
            st.session_state.dni = dni_input
            st.session_state.page = 'Chat'
            # Limpiar cache de recursos al hacer login para recargar modelos/índice si es necesario
            st.cache_resource.clear()
            st.success(f"Bienvenido/a DNI: {dni_input}")
            st.rerun()
        else:
            st.error("DNI o Contraseña incorrectos.")


def change_password_page():
    # (Sin cambios respecto al código anterior)
    st.header("Cambiar Contraseña")
    current_password = st.text_input("Contraseña Actual", type="password")
    new_password = st.text_input("Nueva Contraseña", type="password")
    confirm_password = st.text_input("Confirmar Nueva Contraseña", type="password")
    if st.button("Actualizar Contraseña"):
        # ... (lógica de validación y actualización)
        if not current_password or not new_password or not confirm_password:
            st.warning("Por favor, complete todos los campos.")
            return
        stored_hash = get_user_hash(st.session_state.dni)
        if not verify_password(stored_hash, current_password):
            st.error("La contraseña actual es incorrecta.")
            return
        if new_password != confirm_password:
            st.error("Las nuevas contraseñas no coinciden.")
            return
        if len(new_password) < 4:
             st.warning("La nueva contraseña debe tener al menos 4 caracteres.")
             return
        if update_password(st.session_state.dni, new_password):
            st.success("¡Contraseña actualizada correctamente!")
        else:
            st.error("Ocurrió un error al actualizar la contraseña.")


def admin_page():
    """Página de administración de documentos."""
    st.header("Administración de Documentos")
    st.write("Sube aquí los documentos (PDF, DOCX) que formarán la base de conocimiento del chatbot.")
    st.warning("Al procesar nuevos documentos, se añadirán al conocimiento existente. Para eliminar documentos específicos, actualmente se requiere regenerar la base completa (borrando 'faiss_index/' y volviendo a subir todo).")

    uploaded_files = st.file_uploader(
        "Seleccionar documentos...",
        type=["pdf", "docx", "pptx", "xlsx", "txt"],
        accept_multiple_files=True,
        key="file_uploader" # Key para manejar estado
    )

    if st.button("Procesar Documentos Seleccionados") and uploaded_files:
        # Validar que la API key esté disponible
        if not GOOGLE_API_KEY:
             st.error("No se puede procesar sin la GOOGLE_API_KEY configurada.")
             return
        embeddings_model = get_embeddings_model()
        if not embeddings_model:
             st.error("No se puede procesar sin el modelo de embeddings.")
             return

        with st.spinner("Procesando documentos... Esto puede tardar varios minutos dependiendo del tamaño y cantidad."):
            # 1. Cargar contenido
            docs = load_documents(uploaded_files)
            if not docs:
                 st.error("No se pudo extraer contenido de los archivos seleccionados.")
                 return

            # 2. Dividir en Chunks
            chunks = get_text_chunks(docs)
            if not chunks:
                 st.error("No se pudieron generar fragmentos de texto.")
                 return

            # 3. Crear/Actualizar Vector Store
            vectorstore = create_or_update_vectorstore(chunks, embeddings_model)

            if vectorstore:
                st.success(f"¡Éxito! {len(docs)} documentos procesados, {len(chunks)} fragmentos añadidos/actualizados en la base de conocimiento.")
                # Limpiar la cache del chain para que se regenere con el nuevo índice
                st.cache_resource.clear() # Limpia toda la cache de recursos (modelos, chain)
                st.session_state.pop('qa_chain', None) # Eliminar específicamente el chain del estado
                # Limpiar lista de archivos subidos para evitar reprocesar accidentalmente
                # st.session_state.file_uploader = [] # Esto puede dar error, Streamlit maneja el uploader
                st.rerun() # Forzar recarga para que el chat use el nuevo índice
            else:
                st.error("Ocurrió un error al crear/actualizar la base de conocimiento.")

    # TODO (Opcional): Añadir funcionalidad para listar/eliminar documentos (requiere más lógica)


def chat_page():
    """Página del Chatbot."""
    st.header("Chat de Consultas")

    # Validar que la API key esté disponible
    if not GOOGLE_API_KEY:
         st.error("La funcionalidad del chat no está disponible debido a un problema con la API Key.")
         return

    # Intentar cargar la cadena de QA (usa cache si es posible)
    vectorstore = load_vectorstore()
    qa_chain = None
    if vectorstore:
         # Pasar vectorstore como argumento para que la cache funcione correctamente
        qa_chain = get_qa_chain(vectorstore)

    if not qa_chain:
         st.warning("⚠️ La base de conocimiento no está lista o no se pudo cargar. Por favor, ve a 'Administrar Documentos' y procesa los archivos necesarios.")
         # return # O permitir chatear sin RAG (menos útil)

    st.write(f"Conectado como: DNI {st.session_state.dni}")
    if qa_chain:
        st.info("Escribe tu consulta sobre normativas o procedimientos de RRHH.")
    else:
         st.info("Esperando a que la base de conocimiento esté lista...")


    # Inicializar historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu consulta aquí..." if qa_chain else "Base de conocimiento no disponible..."):
        if not qa_chain:
            st.error("El chatbot no está listo. Por favor, carga documentos en la sección de Administración.")
            return

        # Añadir mensaje del usuario al historial y mostrarlo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Procesar consulta con la cadena de QA
        with st.spinner("Pensando..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                # print("Respuesta completa del chain:", response) # Para depuración

                bot_response = response.get('result', "Lo siento, no pude procesar tu consulta.")

                # Opcional: Mostrar fuentes si se retornaron
                source_documents = response.get('source_documents')
                if source_documents:
                    with st.expander("Ver fuentes consultadas"):
                        for doc in source_documents:
                            st.markdown(f"**Fuente (página {doc.metadata.get('page', 'N/A')}):**")
                            st.caption(doc.page_content[:300] + "...") # Mostrar inicio del chunk

            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")
                bot_response = "Ocurrió un error inesperado al intentar responder."

            # Añadir respuesta del bot al historial y mostrarla
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)

            # Registrar en Log
            add_log(st.session_state.dni, prompt, bot_response)

# --- Flujo Principal de la App ---

def main_app():
    """Muestra la aplicación principal una vez logueado."""
    st.sidebar.header(f"Usuario: {st.session_state.dni}")

    menu = ["Chat", "Cambiar Contraseña", "Administrar Documentos", "Informes Utilizacion"]
    # Usar el índice actual si existe, sino default a 0 (Chat)
    current_page_index = menu.index(st.session_state.page) if st.session_state.page in menu else 0
    st.session_state.page = st.sidebar.radio("Menú", menu, index=current_page_index)

    if st.sidebar.button("Salir"):
        # Limpiar estado de sesión al salir
        keys_to_keep = [] # Mantener claves que no deben borrarse al salir
        for key in list(st.session_state.keys()):
             if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state.logged_in = False
        st.session_state.dni = None
        st.cache_resource.clear() # Limpiar cache al salir
        st.success("Sesión cerrada.")
        st.rerun()

    # Mostrar la página seleccionada
    if st.session_state.page == "Chat":
        chat_page()
    elif st.session_state.page == "Cambiar Contraseña":
        change_password_page()
    elif st.session_state.page == "Administrar Documentos":
        admin_page()
    elif st.session_state.page == "Informes Utilizacion":
        reports_page()


# --- Ejecución ---
if __name__ == "__main__":
    st.set_page_config(page_title="Asistente RRHH", layout="wide") # Configurar título y layout

    # Inicializar DB solo una vez
    if 'db_initialized' not in st.session_state:
        print("Realizando configuración inicial de la DB...")
        init_db()
        st.session_state.db_initialized = True
    else:
        # print("DB ya inicializada.")

        pass

    # Cargar modelos y cadena de QA una vez si el usuario está logueado
    # Esto ahora se maneja dentro de chat_page y admin_page con @st.cache_resource

    # Flujo principal: Login o App
    if not st.session_state.logged_in:
        login_page()
    else:
        # Si está logueado pero falta la API KEY (pudo fallar al inicio)
        if not GOOGLE_API_KEY:
             st.error("La aplicación no puede funcionar correctamente sin la Google API Key.")
             # Mostrar opción de logout o detener
             if st.button("Salir por error de API"):
                 for key in list(st.session_state.keys()):
                     del st.session_state[key]
                     
        # ------ LÍNEA DE DEPURACIÓN ------
        # st.success("DEBUG: Logueado correctamente, intentando mostrar app principal...")
        # print("DEBUG: Logueado correctamente, llamando a main_app()") # También en consola
        # ---------------------------------
        main_app()
