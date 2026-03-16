# Brain Tumor Detection - Weekly Diary (16 Weeks)

## Week 1 - Half 1

### Work Done (Key Progress)

- Finalized minor project topic with guide.
- Defined 4 output classes for detection.
- Chose MRI image based classification approach.
- Compared classical ML and deep learning options.
- Selected SVM + PCA for initial implementation.

### Result Note

Project scope became clear and feasible.
Started with a practical roadmap for semester.

## Week 1 - Half 2

### Work Done

- Created repository and base folder structure.
- Added dataset folders for each class.
- Reviewed available MRI samples manually.
- Drafted initial project flow diagram.
- Prepared software stack list for development.

### Result Note

Foundation setup was completed successfully.
Ready to begin data loading and preprocessing.

## Week 2 - Half 1

### Work Done

- Collected and organized MRI image dataset.
- Mapped classes to numeric labels.
- Checked folder integrity and missing files.
- Verified image formats used in dataset.
- Counted class-wise image distribution.

### Result Note

Dataset became usable for training pipeline.
Basic balance across classes was acceptable.

## Week 2 - Half 2

### Work Done

- Implemented image reading using OpenCV.
- Added grayscale conversion in loading step.
- Added fixed resizing to 200x200.
- Skipped corrupted files safely.
- Stored arrays in NumPy structures.

### Result Note

Raw data pipeline started working end-to-end.
Prepared arrays for model input stage.

## Week 3 - Half 1

### Work Done

- Reshaped image arrays into flat vectors.
- Added normalization by dividing with 255.
- Performed train test split (70/30).
- Fixed random state for reproducibility.
- Verified shapes of train and test sets.

### Result Note

Model input format is now standardized.
Training data prepared for dimensionality reduction.

## Week 3 - Half 2

### Work Done

- Studied feature explosion in pixel space.
- Applied PCA with 98 percent variance.
- Reduced dimensionality significantly.
- Compared memory and speed before/after PCA.
- Exported transformed train/test features.

### Result Note

Computation became lighter and faster.
PCA output ready for classifier training.

## Week 4 - Half 1

### Work Done

- Implemented baseline Logistic Regression model.
- Trained model on PCA-transformed training data.
- Evaluated baseline on test split.
- Recorded initial class-wise behavior.
- Logged baseline metrics for comparison.

### Result Note

Baseline gave a reliable reference point.
Used it to judge SVM improvements.

## Week 4 - Half 2

### Work Done

- Implemented SVM classifier for multiclass task.
- Trained SVM on same PCA features.
- Tested predictions on held-out images.
- Compared outputs with baseline model.
- Selected SVM as primary model.

### Result Note

SVM performance was stable for dataset size.
Primary ML model decision finalized.

## Week 5 - Half 1

### Work Done

- Re-ran training to validate consistency.
- Checked confusion between similar classes.
- Noted edge cases in glioma/pituitary predictions.
- Improved data checks around preprocessing.
- Documented model accuracy observations.

### Result Note

Model quality became repeatable across runs.
Known weak cases were identified clearly.

## Week 5 - Half 2

### Work Done

- Enabled probability mode in SVM.
- Extracted confidence values from predictions.
- Mapped numeric output to class names.
- Built helper function for inference output.
- Verified output schema for web integration.

### Result Note

Inference now returns class + confidence.
Backend function became ready for Flask use.

## Week 6 - Half 1

### Work Done

- Initialized Flask application in app.py.
- Added route for home page.
- Added route for result page.
- Configured upload folder for requests.
- Connected templates folder with Flask rendering.

### Result Note

Web server skeleton became functional.
Routing flow is ready for integration.

## Week 6 - Half 2

### Work Done

- Built first version of home.html.
- Added project title and short description.
- Added file upload form in UI.
- Configured multipart form submission.
- Added simple CSS for readability.

### Result Note

Frontend became usable for manual testing.
Image upload journey started from browser.

## Week 7 - Half 1

### Work Done

- Created result.html template.
- Displayed predicted tumor class in page.
- Added return link to home page.
- Added conditional blocks for result output.
- Tested template rendering with dummy values.

### Result Note

Prediction view pipeline became visible.
User flow improved from input to output.

## Week 7 - Half 2

### Work Done

- Integrated main model file with Flask route.
- Saved uploaded file temporarily on disk.
- Read saved file as binary for model.
- Called prediction function from route.
- Removed temporary file after inference.

### Result Note

End-to-end upload to prediction worked.
Basic privacy cleanup logic was added.

## Week 8 - Half 1

### Work Done

- Added allowed extension validation list.
- Checked empty filename and missing file cases.
- Added error response for invalid uploads.
- Wrapped decode/predict section in try-except.
- Returned clean message on decode failure.

### Result Note

System became safer for user inputs.
Common upload failures are now handled.

## Week 8 - Half 2

### Work Done

- Refactored code into helper functions.
- Separated dataset loading and training blocks.
- Separated uploaded image preprocessing logic.
- Kept backward-compatible prediction helper.
- Simplified app.py route readability.

### Result Note

Codebase became cleaner and maintainable.
Future feature addition became easier.

## Week 9 - Half 1

### Work Done

- Passed confidence score to result template.
- Added confidence percentage display in UI.
- Rounded confidence for clear presentation.
- Tested multiple MRI samples through web app.
- Verified 4-class mapping in final output.

### Result Note

Output now looks more professional.
User gets confidence with predicted class.

## Week 9 - Half 2

### Work Done

- Performed repeated local smoke testing.
- Checked syntax using Python compile step.
- Validated no crashes in main flow.
- Tested unsupported file-type behavior.
- Confirmed temporary file cleanup path.

### Result Note

Core functionality remained stable in tests.
App is reliable for demo usage.

## Week 10 - Half 1

### Work Done

- Faced dependency issue on Python 3.14.
- Investigated scipy/scikit-learn pin conflicts.
- Updated requirements to top-level packages.
- Updated pyproject Python compatibility range.
- Reinstalled dependencies in virtual environment.

### Result Note

Environment setup became smoother on laptop.
Project now installs without old pin issues.

## Week 10 - Half 2

### Work Done

- Ran app startup verification after fixes.
- Confirmed Flask server starts on localhost.
- Verified homepage and upload form rendering.
- Verified prediction endpoint returns result page.
- Finalized current project state for review.

### Result Note

Current implementation is fully working.
Ready for guide presentation and viva.

## Week 11 - Half 1 (Planned)

### Planned Work

- Introduce model save/load with joblib.
- Avoid retraining model on every startup.
- Add cache validity check for model files.
- Add retrain fallback if cache missing.
- Test startup time before and after.

### Expected Note

Startup time should reduce significantly.
Deployment readiness will improve.

## Week 11 - Half 2 (Planned)

### Planned Work

- Add train script separate from app runtime.
- Save PCA and SVM artifacts together.
- Store metadata like class mapping.
- Add version tag for model artifacts.
- Update README with new model flow.

### Expected Note

Training and inference flow will be separated.
Code structure will become production-like.

## Week 12 - Half 1 (Planned)

### Planned Work

- Add optional image enhancement pipeline.
- Try contrast normalization for MRI scans.
- Compare model confidence before/after filter.
- Keep enhancement toggle configurable.
- Validate no distortion in key regions.

### Expected Note

Image quality handling will improve robustness.
Borderline predictions may become better.

## Week 12 - Half 2 (Planned)

### Planned Work

- Add preprocessing logs for debug mode.
- Track image size and processing steps.
- Add graceful handling for unusual dimensions.
- Add tests for non-standard image files.
- Refine error text for user clarity.

### Expected Note

Debugging and troubleshooting will become easier.
User experience will improve in edge cases.

## Week 13 - Half 1 (Planned)

### Planned Work

- Design simple JSON prediction endpoint.
- Accept image via API request payload.
- Return tumor type and confidence in JSON.
- Keep existing HTML route unchanged.
- Validate API input schema strictly.

### Expected Note

Project will support both UI and API usage.
Integration options will increase.

## Week 13 - Half 2 (Planned)

### Planned Work

- Add multi-image batch prediction support.
- Loop through list of uploaded images.
- Return result array with per-image status.
- Handle partial failures gracefully.
- Measure average batch processing time.

### Expected Note

Bulk analysis workflow will become possible.
System utility for labs will increase.

## Week 14 - Half 1 (Planned)

### Planned Work

- Explore basic explainability method options.
- Map important regions influencing prediction.
- Prototype simple heatmap visualization.
- Attach visualization in result page.
- Validate explanation consistency on samples.

### Expected Note

Prediction output will become more interpretable.
Guide discussion quality will improve.

## Week 14 - Half 2 (Planned)

### Planned Work

- Add downloadable result snapshot.
- Include class, confidence, and heatmap.
- Keep report format concise for records.
- Add timestamp and filename metadata.
- Validate exports for multiple samples.

### Expected Note

Result documentation will become easier.
Presentation support artifacts will be ready.

## Week 15 - Half 1 (Planned)

### Planned Work

- Add lightweight database integration.
- Save prediction history per request.
- Store file name, class, confidence, time.
- Add simple history listing page.
- Validate write/read operations locally.

### Expected Note

Prediction records will become trackable.
Project will gain audit capability.

## Week 15 - Half 2 (Planned)

### Planned Work

- Add search by date and class.
- Add deletion option for old records.
- Add basic input validation for history filters.
- Ensure no sensitive image is stored.
- Update privacy note in project docs.

### Expected Note

History module will become usable and safe.
Data handling clarity will improve.

## Week 16 - Half 1 (Planned)

### Planned Work

- Prepare deployment-ready app config.
- Disable debug mode for production run.
- Add environment variable based settings.
- Test run with gunicorn locally.
- Document deployment command steps.

### Expected Note

Project will be ready for hosted deployment.
Runtime behavior will be more stable.

## Week 16 - Half 2 (Planned)

### Planned Work

- Perform final end-to-end regression tests.
- Recheck all routes and edge cases.
- Clean code and remove unused sections.
- Prepare final PPT and live demo flow.
- Freeze final submission package.

### Expected Note

Minor project closure will be complete.
System will be ready for evaluation.
