# Etapa 1: Build
FROM node:18 AS builder

# Set working directory
WORKDIR /app

# Copiar archivos de dependencias
COPY package.json package-lock.json* ./

# Instalación con legacy-peer-deps
RUN npm install --legacy-peer-deps \
    && npm install react-markdown remark-gfm rehype-raw --legacy-peer-deps\
    && npm install react-syntax-highlighter @types/react-syntax-highlighter --legacy-peer-deps

# Copiar el resto del proyecto
COPY . .
# Build del proyecto Next.js
RUN npm run build

# Etapa 2: Producción
FROM node:18-alpine AS runner

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios desde el build
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Exponer el puerto por defecto de Next.js
EXPOSE 3000

# Comando de inicio
CMD ["npx", "next", "start", "-H", "0.0.0.0"]
