# Viralify PPTX Service

Production-ready Node.js microservice for PowerPoint generation using PptxGenJS.

## Features

- **PPTX Generation**: Generate professional PowerPoint presentations from JSON
- **Syntax Highlighting**: Code slides with language-specific syntax highlighting
- **Themes**: 7 built-in themes (dark, light, corporate, gradient, ocean, neon, minimal)
- **Transitions**: Slide transitions (fade, push, wipe, zoom, split, reveal, cover)
- **PNG Export**: Convert slides to PNG images via LibreOffice
- **Async Jobs**: Async generation with status polling
- **Charts & Tables**: Support for charts (bar, line, pie) and tables

## Port

- **8013** (configurable via `PORT` env)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with LibreOffice status |
| `/api/v1/pptx/generate` | POST | Generate PPTX (sync) |
| `/api/v1/pptx/generate-async` | POST | Generate PPTX (async) |
| `/api/v1/pptx/preview` | POST | Generate single slide PNG |
| `/api/v1/pptx/jobs/:id` | GET | Get job status |
| `/api/v1/pptx/files/:file` | GET | Download generated file |
| `/api/v1/pptx/themes` | GET | List available themes |
| `/api/v1/pptx/slide-types` | GET | List slide types |
| `/api/v1/pptx/cleanup` | POST | Clean up old files |

## Slide Types

- `title` - Title slide with heading and subtitle
- `content` - Generic content slide
- `code` / `code_demo` - Code with syntax highlighting
- `diagram` - Architecture/diagram slides
- `comparison` - Side-by-side comparison
- `two_column` - Two-column layout
- `bullet_points` - Bullet points list
- `quote` - Quote/testimonial
- `image` - Full image slide
- `conclusion` - Conclusion with key takeaways
- `section_header` - Section divider

## Usage Example

```json
POST /api/v1/pptx/generate
{
  "job_id": "course_123",
  "slides": [
    {
      "type": "title",
      "title": "Introduction to Apache Kafka",
      "subtitle": "Distributed Streaming Platform"
    },
    {
      "type": "code",
      "title": "Producer Example",
      "codeBlocks": [
        {
          "language": "python",
          "code": "from kafka import KafkaProducer\n\nproducer = KafkaProducer(bootstrap_servers='localhost:9092')\nproducer.send('my-topic', b'Hello, Kafka!')"
        }
      ]
    }
  ],
  "theme": {
    "style": "dark"
  },
  "outputFormat": "both",
  "metadata": {
    "title": "Kafka Course",
    "author": "Viralify"
  }
}
```

## Response

```json
{
  "success": true,
  "job_id": "course_123",
  "pptx_url": "/api/v1/pptx/files/course_123_1234567890.pptx",
  "png_urls": [
    "/api/v1/pptx/files/abc123/slide-1.png",
    "/api/v1/pptx/files/abc123/slide-2.png"
  ],
  "processing_time_ms": 1250
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8013 | Service port |
| `NODE_ENV` | development | Environment |
| `LOG_LEVEL` | info | Logging level |
| `PPTX_OUTPUT_DIR` | /tmp/viralify/pptx | PPTX output directory |
| `PPTX_IMAGES_DIR` | /tmp/viralify/pptx-images | PNG output directory |
| `LIBREOFFICE_PATH` | auto-detect | Path to LibreOffice |

## Docker

```bash
# Build
docker build -t viralify-pptx-service .

# Run
docker run -p 8013:8013 \
  -v pptx_output:/tmp/viralify/pptx \
  -v pptx_images:/tmp/viralify/pptx-images \
  viralify-pptx-service
```

## Dependencies

- **PptxGenJS**: PowerPoint generation
- **LibreOffice**: PPTX → PNG conversion (optional)
- **poppler-utils**: PDF → PNG conversion (pdftoppm)
- **ImageMagick**: Fallback for PNG conversion

## Integration with Viralify

The service integrates with `presentation-generator` via the `pptx_client.py` module:

```python
from services.pptx_client import PptxClient, Slide, SlideType, PresentationTheme

client = PptxClient()

result = await client.generate(
    job_id="course_123",
    slides=[
        Slide(type=SlideType.TITLE, title="Hello World"),
        Slide(type=SlideType.CODE, code_blocks=[...]),
    ],
    theme=PresentationTheme(style=ThemeStyle.DARK),
    output_format="both",
)

if result.success:
    print(f"PPTX: {result.pptx_url}")
    print(f"PNGs: {result.png_urls}")
```

## License

MIT
