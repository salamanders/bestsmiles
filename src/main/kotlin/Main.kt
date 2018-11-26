import mu.KotlinLogging
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Rectangle
import java.nio.file.Paths


private val logger = KotlinLogging.logger {}

fun findStuff(net: Net, filename: String): List<Pair<Rectangle, String>> {

    val frame = Imgcodecs.imread(filename)!!
    val frameWidth = frame.cols()
    val frameHeight = frame.rows()

    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB) // Strip out any alpha

    // TODO: This looks like it needs plenty of knowledge.
    //val blob = Dnn.blobFromImage(frame, 1.0, Size(IN_WIDTH, IN_HEIGHT), Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false)
    val blob = Dnn.blobFromImage(frame)!!
    logger.info { "Made blob." }

    net.setInput(blob)
    val detectionsUnshaped = net.forward()
    detectionsUnshaped.reshape(1, detectionsUnshaped.total().toInt() / 7)!!.let { detections ->
        logger.info { "Got detections: ${detections.rows()}" }
        return (0 until detections.rows()).mapNotNull { i ->
            if (detections.get(i, 2)[0] < 0.2) {
                null
            } else {
                val classId = detections.get(i, 1)[0].toInt()
                val left = (detections.get(i, 3)[0] * frameWidth).toInt()
                val top = (detections.get(i, 4)[0] * frameHeight).toInt()
                val right = (detections.get(i, 5)[0] * frameWidth).toInt()
                val bottom = (detections.get(i, 6)[0] * frameHeight).toInt()
                // Draw rectangle around detected object.
                Imgproc.rectangle(frame, Point(left.toDouble(), top.toDouble()), Point(right.toDouble(), bottom.toDouble()),
                        Scalar(0.0, 255.0, 0.0))
                Pair(Rectangle(left, top, right - left, bottom - top), "face")
            }
        }
    }
}


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

    // Get a new frame
    val faces = findStuff(net, "DSC02953.JPG")
    println(faces.joinToString())
}