using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.IO;
using System.Linq;
using System.Windows;

namespace FaceRecognition
{
    /// <summary>
    /// Логика взаимодействия для FaceRegistrationWindow.xaml
    /// </summary>
    public partial class FaceRegistrationWindow : Window
    {
        public string FaceName { get; set; } = "Безымянное лицо";
        public FaceData[] Faces { get; }
        public FaceData SelectedFace { get; set; }
        public FaceRecognizer FaceRecognizer { get; set; }

        public FaceRegistrationWindow(FaceData[] faces)
        {
            Faces = faces;

            DataContext = this;
            InitializeComponent();
        }

        private void SaveFace(object sender, RoutedEventArgs e)
        {
            if (SelectedFace == null)
            {
                MessageBox.Show("Выберите лицо для регистрации", "Ошибка регистрации", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            FaceRecognizer = new FisherFaceRecognizer(0, 3500);
            var imageList = new VectorOfMat();
            var labelList = new VectorOfInt();

            imageList.Push(SelectedFace.CVImage.Resize(100, 100, Inter.Cubic).Mat);

            var samples = Directory.GetFiles("Face Samples");
            foreach (var sample in samples)
            {
                imageList.Push(new Image<Gray, byte>(sample).Resize(100, 100, Inter.Cubic).Mat);
            }

            labelList.Push(Enumerable.Range(1, samples.Length + 1).ToArray());
            FaceRecognizer.Train(imageList, labelList);

            DialogResult = true;
            Close();
        }
    }
}
