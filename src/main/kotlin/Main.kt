import mu.KotlinLogging
import org.opencv.core.Core
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.dnn.Dnn
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.nio.file.Paths


private val logger = KotlinLogging.logger {}


// https://github.com/opencv/opencv/blob/master/samples/android/mobilenet-objdetect/src/org/opencv/samples/opencv_mobilenet/MainActivity.java
fun main() {
    logger.info { "Starting app." }

    nu.pattern.OpenCV.loadShared() // Required first by org.openpnp
    System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME)

    val prototxt = Paths.get("src/main/resources/MobileNetSSD_deploy.prototxt.txt")!!
    val caffeModel = Paths.get("src/main/resources/MobileNetSSD_deploy.caffemodel")!!

    require(prototxt.toFile().canRead())
    require(caffeModel.toFile().canRead())

    val net = Dnn.readNetFromCaffe(prototxt.toString(), caffeModel.toString())!!

    //val net = Dnn.readNetFromCaffe("/Users/benhill/Documents/workspace/bestsmiles/MobileNetSSD_deploy.prototxt", "/Users/benhill/Documents/workspace/bestsmiles/mobilenet_iter_73000.caffemodel")!!
    logger.info { "Network loaded successfully" }

    val IN_SCALE_FACTOR = 0.007843
    val MEAN_VAL = 127.5
    val THRESHOLD = 0.2
    // Get a new frame


    val frame = Imgcodecs.imread("DSC02953.JPG")!!

    //Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)
    // Forward image through network.

    val blob = Dnn.blobFromImage(frame)!!
    logger.info { "Made blob." }

    //val blob = Dnn.blobFromImage(frame, 1.0, Size(IN_WIDTH, IN_HEIGHT), Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false)
    net.setInput(blob)
    val detectionsUnshaped = net.forward()
    val cols = frame.cols()
    val rows = frame.rows()
    val detections = detectionsUnshaped.reshape(1, detectionsUnshaped.total().toInt() / 7)!!
    logger.info { "Got detections: ${detections.rows()}" }
    for (i in 0 until detections.rows()) {
        val confidence = detections.get(i, 2)[0]
        if (confidence > THRESHOLD) {
            val classId = detections.get(i, 1)[0].toInt()
            val left = (detections.get(i, 3)[0] * cols).toInt()
            val top = (detections.get(i, 4)[0] * rows).toInt()
            val right = (detections.get(i, 5)[0] * cols).toInt()
            val bottom = (detections.get(i, 6)[0] * rows).toInt()
            logger.info { "Detection $i: ($left,$top) = $classId:$confidence" }
            // Draw rectangle around detected object.
            Imgproc.rectangle(frame, Point(left.toDouble(), top.toDouble()), Point(right.toDouble(), bottom.toDouble()),
                    Scalar(0.0, 255.0, 0.0))
            val label = "$classId:$confidence"
            val baseLine = IntArray(1)
            val labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine)
            // Draw background for label.
            Imgproc.rectangle(frame, Point(left.toDouble(), top - labelSize.height),
                    Point(left + labelSize.width, top + baseLine[0].toDouble()),
                    Scalar(255.0, 255.0, 255.0))
            // Write class name and confidence.
            Imgproc.putText(frame, label, Point(left.toDouble(), top.toDouble()),
                    Core.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0.0, 0.0, 0.0))
        }
    }
}