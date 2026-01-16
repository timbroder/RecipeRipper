# CLAUDE_APP.md - RecipeRipper Mobile

This file provides context for AI assistants working on the mobile app version of this project.

## Project Overview

RecipeRipper Mobile is a Flutter application that extracts structured recipes from cooking videos. It's a mobile conversion of the original Python CLI tool, designed to run entirely on-device using native iOS and Android ML capabilities.

**Original Project**: https://github.com/timbroder/RecipeRipper (Python CLI)
**Mobile Project**: https://github.com/timbroder/RecipeRipperApp (Flutter)

## Core Functionality

- Accept video URLs via share sheet or pick local videos from camera roll
- Download and preview videos before processing
- Extract audio and transcribe using on-device speech recognition
- Extract on-screen text using OCR from video frames
- Parse transcription + OCR text into structured recipes (ingredients + directions)
- Store recipes locally with optional cloud sync
- Allow full editing of extracted recipes
- Export recipes as JSON/Markdown
- Process in background with notifications
- Delete videos after processing to save space

## Tech Stack

### Framework & Languages
- **Flutter 3.x+** (Dart 3.x+)
- **iOS**: Swift for native Speech + Vision frameworks
- **Android**: Kotlin for native Speech Recognition + ML Kit

### Key Dependencies
- `sqflite` - Local SQLite database
- `image_picker` / `file_picker` - Video selection
- `ffmpeg_kit_flutter` - Video/audio processing
- `video_player` - Video playback and thumbnail generation
- `shared_preferences` - App settings
- `path_provider` - File system access

### Native Integrations
- **iOS**: Speech framework (transcription), Vision framework (OCR)
- **Android**: SpeechRecognizer API, ML Kit Text Recognition
- **iOS**: Background Tasks framework, CloudKit/iCloud Drive
- **Android**: WorkManager, Google Drive API

## Architecture

```
lib/
├── main.dart                 # App entry point
├── models/                   # Data models (Recipe, Ingredient, Direction)
├── services/                 # Business logic
│   ├── video_service.dart    # Download, extract frames
│   ├── speech_service.dart   # Platform channel for transcription
│   ├── ocr_service.dart      # Platform channel for OCR
│   ├── parsing_service.dart  # Recipe parsing heuristics
│   ├── database_service.dart # SQLite operations
│   └── sync_service.dart     # Cloud sync
├── screens/                  # UI screens
│   ├── home_screen.dart      # Recipe grid
│   ├── detail_screen.dart    # Recipe view
│   └── edit_screen.dart      # Recipe editing
├── widgets/                  # Reusable components
└── utils/                    # Helpers and constants

ios/
├── Runner/
│   ├── AppDelegate.swift
│   ├── SpeechBridge.swift   # Native speech transcription
│   └── VisionBridge.swift   # Native OCR

android/
├── app/src/main/kotlin/
│   ├── MainActivity.kt
│   ├── SpeechBridge.kt      # Native speech transcription
│   └── OcrBridge.kt         # Native OCR
```

## Development Workflow

### Setup
```bash
flutter pub get
flutter run
```

### Testing
```bash
flutter test                    # Unit tests
flutter test integration_test/  # Integration tests
```

### Building
```bash
flutter build ios
flutter build apk
flutter build appbundle
```

## Key Design Decisions

### Why On-Device ML?
- No API costs
- Works offline
- Privacy (no data leaves device)
- Faster (no network latency)

### Why Delete Videos After Processing?
- Mobile storage is limited
- Users need recipes, not videos
- Can always re-download from source
- Reduces app footprint

### Why Flutter?
- Single codebase for iOS + Android
- Native performance for video/ML workloads
- Excellent plugin ecosystem
- Fast development with hot reload

## Parsing Heuristics (Ported from Python)

### Ingredient Detection
- Has quantities/units (1 cup, 2 tbsp, etc.)
- Contains common ingredient keywords (flour, sugar, salt, etc.)
- Shorter text (ingredients are concise)
- Often bullet-pointed or listed

### Direction Detection
- Imperative verbs (bake, mix, pour, stir, etc.)
- Contains temperatures (350°F, 180°C)
- Contains times (10 minutes, 2 hours)
- Numbered steps
- Longer, sentence-form text

### Normalization
- Convert unit abbreviations (tbsp→tablespoon, tsp→teaspoon)
- Handle fractions (1/2, ¼, ½)
- Deduplicate similar ingredients
- Clean up OCR artifacts

## Testing Strategy

### Unit Tests (>90% coverage target)
- All parsing functions
- Unit conversions
- Ingredient/direction classification
- Text normalization

### Widget Tests
- All screens render correctly
- User interactions work
- State management updates UI

### Integration Tests
- Share URL → Process → View recipe (end-to-end)
- Pick local video → Edit recipe → Export
- Background processing with notifications

### Manual Testing Checklist
See PROJECT_PLAN.md for comprehensive test plan

## Common Commands

```bash
# Run on specific device
flutter run -d <device-id>

# Run tests with coverage
flutter test --coverage

# Build release
flutter build ios --release
flutter build appbundle --release

# Clean build
flutter clean && flutter pub get

# Format code
dart format .

# Analyze code
flutter analyze
```

## Platform-Specific Notes

### iOS
- Requires iOS 14+ for Speech framework offline support
- Add privacy descriptions in Info.plist:
  - `NSSpeechRecognitionUsageDescription`
  - `NSMicrophoneUsageDescription` (even though we don't use mic live)
  - `NSPhotoLibraryUsageDescription`
- Enable Background Modes capability (Audio, Processing)
- Enable iCloud capability for sync

### Android
- Min SDK 26 (Android 8.0) for Speech Recognition
- Add permissions in AndroidManifest.xml:
  - `RECORD_AUDIO`
  - `READ_EXTERNAL_STORAGE`
  - `INTERNET` (for URL downloads)
- Configure WorkManager for background tasks
- Set up Google Drive API credentials for sync

## Known Limitations

- YouTube downloading may violate TOS (focus on local videos)
- Processing long videos (>30 min) may drain battery
- OCR accuracy depends on video quality and text clarity
- Transcription accuracy depends on audio quality and accents
- Background processing may be killed by OS under extreme memory pressure

## Troubleshooting

### "No devices found"
```bash
flutter devices
# For iOS: open Xcode, Window → Devices and Simulators
# For Android: adb devices
```

### Platform channel not found
- Make sure native code is properly registered in AppDelegate/MainActivity
- Rebuild app completely: `flutter clean && flutter run`

### Speech/OCR not working
- Check permissions are granted
- Verify native code is calling correct APIs
- Check device language settings
- Ensure models are downloaded (iOS Vision, Android ML Kit)

## Resources

- [Flutter Documentation](https://docs.flutter.dev/)
- [iOS Speech Framework](https://developer.apple.com/documentation/speech)
- [iOS Vision Framework](https://developer.apple.com/documentation/vision)
- [Android Speech Recognition](https://developer.android.com/reference/android/speech/SpeechRecognizer)
- [ML Kit Text Recognition](https://developers.google.com/ml-kit/vision/text-recognition)
- [Original Python Project](https://github.com/timbroder/RecipeRipper)

## Sprint Plan

See PROJECT_PLAN.md for detailed sprint breakdown (Sprints 0-6, ~14-15 weeks to MVP)

## Contributing

When working on this project:
1. Read the sprint plan to understand current phase
2. Follow Flutter style guide
3. Write tests for new features
4. Update this document if architecture changes
5. Test on both iOS and Android before committing
6. Keep native code minimal (prefer Dart when possible)

## Questions?

Check the original Python implementation for parsing logic reference:
- `recipe_extractor.py` - Core parsing heuristics
- `tests/test_recipe_extractor.py` - Test cases showing expected behavior

---

**Last Updated**: 2026-01-16
**Current Phase**: Planning (Pre-Sprint 0)
