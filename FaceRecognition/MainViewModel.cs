using Caliburn.Micro;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Timers;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace FaceRecognition
{
    class MainViewModel : PropertyChangedBase
    {
        private readonly VideoCapture _videoCapture;
        private readonly CascadeClassifier _haarCascade;
        private FaceRecognizer _faceRecognizer;

        private readonly Timer _updateTimer;
        private readonly Timer _workTimer;
        private readonly Timer _timeoutTimer;

        private const int WorkTime = 20, TimeoutTime = 15, UpdateInterval = 1;
        private int _elapsedWorkSec, _elapsedTimeoutSec;

        private string _userName;
        private bool _isUserDetected;

        private BitmapImage _cameraImage;
        public BitmapImage CameraImage
        {
            get => _cameraImage;
            set
            {
                if (_cameraImage != value)
                {
                    _cameraImage = value;
                    NotifyOfPropertyChange(() => CameraImage);
                }
            }
        }

        private FaceData[] _capturedFaces;
        public FaceData[] CapturedFaces
        {
            get => _capturedFaces;
            set
            {
                if (_capturedFaces != value)
                {
                    _capturedFaces = value;
                    NotifyOfPropertyChange(() => CapturedFaces);
                }
            }
        }

        private string _state = "Для начала работы зарегистрируйте лицо";
        public string State
        {
            get => _state;
            set
            {
                if (_state != value)
                {
                    _state = value;
                    NotifyOfPropertyChange(() => State);
                }
            }
        }

        public bool IsFaceRegistered { get; set; } = false;

        public ICommand RegisterFaceCommand => new RelayCommand(s =>
            {
                _updateTimer.Stop();
                var registerWindow = new FaceRegistrationWindow(CapturedFaces);
                if (registerWindow.ShowDialog() == true)
                {
                    IsFaceRegistered = true;
                    NotifyOfPropertyChange(() => IsFaceRegistered);

                    _faceRecognizer = registerWindow.FaceRecognizer;
                    _userName = registerWindow.FaceName;
                    _workTimer.Start();
                }
                _updateTimer.Start();
            });

        public MainViewModel()
        {
            _haarCascade = new CascadeClassifier(@"Resources\haarcascade_frontalface_default.xml");

            _videoCapture = new VideoCapture();
            _videoCapture.SetCaptureProperty(CapProp.Fps, 10);
            _videoCapture.SetCaptureProperty(CapProp.FrameHeight, 1280);
            _videoCapture.SetCaptureProperty(CapProp.FrameWidth, 720);

            _workTimer = new Timer
            {
                Interval = UpdateInterval * 1000,
                AutoReset = true
            };
            _workTimer.Elapsed += _workTimer_Elapsed;

            _timeoutTimer = new Timer
            {
                Interval = UpdateInterval * 1000,
                AutoReset = true
            };
            _timeoutTimer.Elapsed += _timeoutTimer_Elapsed;

            _updateTimer = new Timer
            {
                Interval = 200,
                AutoReset = true
            };
            _updateTimer.Elapsed += _updateTimer_Elapsed;
            _updateTimer.Start();
        }

        private void _timeoutTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (!_isUserDetected)
            {
                _elapsedTimeoutSec -= UpdateInterval;
                State = $"Оставшееся время отдыха: {_elapsedTimeoutSec} сек";

                if (_elapsedTimeoutSec <= 0)
                {
                    _timeoutTimer.Stop();
                    _elapsedWorkSec = 0;
                    _workTimer.Start();
                }
            }
            else
            {
                State = "Работник не отдыхает";
            }
        }

        private void _workTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (_isUserDetected)
            {
                _elapsedWorkSec += UpdateInterval;
                State = $"Время работы: {_elapsedWorkSec} сек";
                
                if (_elapsedWorkSec >= WorkTime)
                {
                    _workTimer.Stop();
                    _elapsedTimeoutSec = TimeoutTime;
                    _timeoutTimer.Start();
                }
            }
            else
            {
                State = "Работник не обнаружен";
            }
        }

        private void _updateTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            var frame = _videoCapture.QueryFrame().ToImage<Bgr, byte>();
            if (frame != null)
            {
                var grayFrame = frame.Convert<Gray, byte>();

                var faces = _haarCascade.DetectMultiScale(grayFrame);
                var capturedFaces = new List<FaceData>();
                var isUserDetected = false;
                foreach (var face in faces)
                {
                    frame.Draw(face, new Bgr(0, 0, 255), 2);
                    var detectedFace = grayFrame.Copy(face);

                    var label = "Безымянное лицо";
                    if (_faceRecognizer != null)
                    {
                        var result = _faceRecognizer.Predict(detectedFace.Resize(100, 100, Inter.Cubic));
                        if (result.Label == 1)
                        {
                            label = _userName;
                            isUserDetected = true;
                        }
                    }

                    var capturedFace = new FaceData
                    {
                        ActualImage = BitmapToImageSource(detectedFace.ToBitmap()),
                        Label = label,
                        CVImage = detectedFace
                    };

                    capturedFaces.Add(capturedFace);
                }
                _isUserDetected = isUserDetected;
                CapturedFaces = capturedFaces.ToArray();
                CameraImage = BitmapToImageSource(frame.ToBitmap());
            }
        }

        private BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                var bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();
                bitmapimage.Freeze();

                return bitmapimage;
            }
        }

        public void Closing()
        {
            _updateTimer.Stop();
            _videoCapture.Dispose();
            _haarCascade.Dispose();
            _faceRecognizer?.Dispose();
        }
    }
}
